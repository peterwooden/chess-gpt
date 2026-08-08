import type { Chess, Color, PieceSymbol, Square } from "chess.js";

export type GameResult = "1-0" | "0-1" | "1/2-1/2";

export type GamePlayer = {
  name: string;
  newGame?(seed: number): Promise<void>;
  predict(
    history: string[],
    legalMoves: string[],
    moveTimeLimitMs: number,
  ): Promise<{ san: string }>;
};

export type PlayedMove = {
  ply: number;
  san: string;
  color: Color;
  from: Square;
  to: Square;
  captured: PieceSymbol | null;
  elapsedMs: number;
  actor: string;
};

export type PlayGameOptions = {
  moveTimeLimitMs: number;
  seed?: number;
  now?: () => number;
  onMove?: (move: PlayedMove, game: Chess) => void | Promise<void>;
  signal?: AbortSignal;
  /**
   * Book moves the runner plays before either package moves. They cost no
   * thinking time, are recorded with actor "book", and reach the packages as
   * ordinary history. An illegal book move throws rather than forfeiting.
   */
  openingMoves?: readonly string[];
  /** Recorded as the PGN Opening header when present. */
  openingName?: string;
};

export type GameOutcome = {
  result: GameResult;
  termination: string;
  moves: PlayedMove[];
  pgn: string;
  failure: { color: Color; reason: string } | null;
};

export function playGame(
  white: GamePlayer,
  black: GamePlayer,
  options: PlayGameOptions,
): Promise<GameOutcome>;
