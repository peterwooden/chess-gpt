import type { Color, Square } from "chess.js";
import { createLiveGameEventSequencer } from "../../lib/live-game-events.mjs";
import type {
  LiveGamePhase,
  LiveGameResult,
  LiveGameSource,
} from "../../lib/live-game-types";
import type { ThinkingSample } from "../../lib/thinking-events.mjs";

const PUBLISH_TIMEOUT_MS = 1_500;
const BATCH_WINDOW_MS = 500;

type LiveGameDescriptor = {
  id: string;
  source: LiveGameSource;
  tournamentId?: string;
  tournamentPairKey?: string;
  tournamentGameIndex?: number;
  runnerId?: string;
  whiteName: string;
  blackName: string;
  whiteModelReference?: string | null;
  blackModelReference?: string | null;
  openingName?: string | null;
};

export type LiveGameUpdate = {
  phase: LiveGamePhase;
  status: string;
  moves: readonly string[];
  lastMoveMs?: number | null;
  result?: LiveGameResult | null;
};

export type LiveGamePublisher = {
  id: string;
  watchPath: string;
  startTurn(ply: number, color: Color, turnId?: string): string;
  thinking(turnId: string, sample: ThinkingSample): void;
  movePlayed(turnId: string | null, move: {
    ply: number;
    san: string;
    color: Color;
    from: Square;
    to: Square;
    actor: string;
    elapsedMs: number;
  }): void;
  publish(update: LiveGameUpdate): Promise<void>;
  dispose(): void;
};

export async function openLiveGamePublisher(
  descriptor: LiveGameDescriptor,
): Promise<LiveGamePublisher> {
  const publisherToken = randomToken();
  const response = await fetchWithTimeout("/api/live-games", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...descriptor, publisherToken }),
  });
  const payload = await response.json() as { error?: string };
  if (!response.ok) throw new Error(payload.error ?? "The live game could not be opened.");

  const sequencer = createLiveGameEventSequencer(() => performance.now());
  sequencer.record({ type: "game.started" });
  let revision = 0;
  let latestUpdate: LiveGameUpdate = {
    phase: "starting",
    status: "Preparing game…",
    moves: [],
  };
  let activeTurn: { id: string; offsetMs: number } | null = null;
  const backlog: Array<{ revision: number; update: LiveGameUpdate; eventBatch: unknown }> = [];
  let queue = Promise.resolve();
  let timer = window.setInterval(() => void flush().catch(() => {}), BATCH_WINDOW_MS);

  return {
    id: descriptor.id,
    watchPath: `/watch/${encodeURIComponent(descriptor.id)}`,
    startTurn(ply, color, turnId) {
      const id = turnId ?? `${ply}-${crypto.randomUUID()}`;
      activeTurn = { id, offsetMs: sequencer.currentOffsetMs() };
      sequencer.record({ type: "turn.started", turnId: id, ply, color });
      return id;
    },
    thinking(turnId, sample) {
      if (!activeTurn || activeTurn.id !== turnId) return;
      sequencer.record(
        { type: "thinking.command", turnId, command: sample.command },
        activeTurn.offsetMs + sample.elapsedMs,
      );
    },
    movePlayed(turnId, move) {
      sequencer.record({ type: "move.played", turnId, ...move });
      if (turnId === activeTurn?.id) activeTurn = null;
    },
    publish(update) {
      latestUpdate = copyUpdate(update);
      sequencer.record({ type: "game.updated", update: latestUpdate });
      if (update.phase === "finished") {
        window.clearInterval(timer);
        timer = 0;
      }
      return flush();
    },
    dispose() {
      window.clearInterval(timer);
      timer = 0;
    },
  };

  function flush(): Promise<void> {
    const eventBatch = sequencer.flush();
    if (eventBatch) {
      revision += 1;
      backlog.push({ revision, update: copyUpdate(latestUpdate), eventBatch });
    }
    if (backlog.length === 0) return queue;
    const operation = queue.then(drain);
    queue = operation.catch(() => {});
    return operation;
  }

  async function drain(): Promise<void> {
    while (backlog.length > 0) {
      const next = backlog[0];
      const updateResponse = await fetchWithTimeout(
        `/api/live-games/${encodeURIComponent(descriptor.id)}`,
        {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            ...next.update,
            publisherToken,
            revision: next.revision,
            eventBatch: next.eventBatch,
          }),
        },
      );
      if (!updateResponse.ok) {
        const updatePayload = await updateResponse.json() as { error?: string };
        throw new Error(updatePayload.error ?? "The live game could not be updated.");
      }
      backlog.shift();
    }
  }
}

function copyUpdate(update: LiveGameUpdate): LiveGameUpdate {
  return { ...update, moves: [...update.moves] };
}

async function fetchWithTimeout(input: RequestInfo | URL, init: RequestInit): Promise<Response> {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), PUBLISH_TIMEOUT_MS);
  try {
    return await fetch(input, { ...init, signal: controller.signal });
  } finally {
    window.clearTimeout(timer);
  }
}

function randomToken(): string {
  const bytes = crypto.getRandomValues(new Uint8Array(32));
  return Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
}
