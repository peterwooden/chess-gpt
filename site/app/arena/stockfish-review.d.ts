export type ReviewJudgement = "inaccuracy" | "mistake" | "blunder";

export type MoveReview = {
  ply: number;
  color: "w" | "b";
  accuracy: number;
  winningChanceLoss: number;
  judgement: ReviewJudgement | null;
  bestMoveSan: string | null;
};

export type PlayerReview = {
  accuracy: number;
  counts: Record<ReviewJudgement, number>;
};

export type GameReview = {
  engine: string;
  nodesPerPosition: number;
  moves: MoveReview[];
  players: { w: PlayerReview; b: PlayerReview };
};

export type ReviewProgress = { completed: number; total: number };

export const STOCKFISH_NODES_PER_POSITION: number;
export const STOCKFISH_ENGINE_NAME: string;
export function analyzeGameWithStockfish(
  sanHistory: readonly string[],
  onProgress?: (progress: ReviewProgress) => void,
  signal?: AbortSignal,
): Promise<GameReview>;
export function buildGameReview(
  evaluations: Array<{ whiteScore: number; bestMoveSan?: string | null }>,
): GameReview;
export function winPercentFromCentiPawns(centiPawns: number): number;
export function moveAccuracy(before: number, after: number): number;
export function classifyWinningChanceLoss(loss: number): ReviewJudgement | null;
