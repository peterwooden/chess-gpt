import type { Square } from "chess.js";

export type ThinkingCommand =
  | { type: "highlightSquare"; square: Square; intensity: number; fadeMs: number }
  | { type: "drawArrow"; from: Square; to: Square; intensity: number; fadeMs: number; side?: "own" | "opponent" }
  | { type: "clearSquare"; square: Square }
  | { type: "clearArrow"; from: Square; to: Square }
  | { type: "clearAll" };

export type ThinkingSample = {
  elapsedMs: number;
  command: ThinkingCommand;
};

export function normalizeThinkingCommand(value: unknown): ThinkingCommand | null;
export function createThinkingCommandLimiter(now: () => number): { accept(): boolean };
