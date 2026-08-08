import type { Color, Square } from "chess.js";
import type { LiveGamePhase, LiveGameResult } from "./live-game-types";
import type { ThinkingCommand } from "./thinking-events.mjs";

export type LiveGameStreamUpdate = {
  phase: LiveGamePhase;
  status: string;
  moves: readonly string[];
  lastMoveMs?: number | null;
  result?: LiveGameResult | null;
};

export type LiveGameEventPayload =
  | { type: "game.started" }
  | { type: "turn.started"; turnId: string; ply: number; color: Color }
  | { type: "thinking.command"; turnId: string; command: ThinkingCommand }
  | {
      type: "move.played";
      turnId: string | null;
      ply: number;
      san: string;
      color: Color;
      from: Square;
      to: Square;
      actor: string;
      elapsedMs: number;
    }
  | { type: "game.updated"; update: LiveGameStreamUpdate };

export type LiveGameEvent = {
  seq: number;
  offsetMs: number;
  payload: LiveGameEventPayload;
};

export type LiveGameEventBatch = {
  batchIndex: number;
  firstSeq: number;
  lastSeq: number;
  events: LiveGameEvent[];
};

export function createLiveGameEventSequencer(now: () => number): {
  currentOffsetMs(): number;
  record(payload: LiveGameEventPayload, capturedOffsetMs?: number): LiveGameEvent | null;
  flush(): LiveGameEventBatch | null;
};

export function normalizeLiveGameEventBatch(value: unknown): LiveGameEventBatch | null;
