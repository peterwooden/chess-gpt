import type {
  LiveGamePhase,
  LiveGameResult,
  LiveGameSource,
} from "../../lib/live-game-types";

const PUBLISH_TIMEOUT_MS = 1_500;

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
  publish(update: LiveGameUpdate): Promise<void>;
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

  let revision = 0;
  let queue = Promise.resolve();
  return {
    id: descriptor.id,
    watchPath: `/watch/${encodeURIComponent(descriptor.id)}`,
    publish(update) {
      revision += 1;
      const nextRevision = revision;
      const operation = queue.then(async () => {
        const updateResponse = await fetchWithTimeout(
          `/api/live-games/${encodeURIComponent(descriptor.id)}`,
          {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              ...update,
              moves: [...update.moves],
              publisherToken,
              revision: nextRevision,
            }),
          },
        );
        if (!updateResponse.ok) {
          const updatePayload = await updateResponse.json() as { error?: string };
          throw new Error(updatePayload.error ?? "The live game could not be updated.");
        }
      });
      queue = operation.catch(() => {});
      return operation;
    },
  };
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

