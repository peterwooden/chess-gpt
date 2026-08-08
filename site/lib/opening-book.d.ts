export type Opening = {
  eco: string;
  name: string;
  moves: string[];
};

export const OPENING_BOOK: ReadonlyArray<Opening>;

export function openingPool(): Opening[];
export function sampleOpenings(count: number, random: () => number): Opening[];
export function openingForSlot(
  openings: ReadonlyArray<Opening> | null,
  gameIndex: number,
): Opening | null;
export function parseOpenings(raw: string | null | undefined): Opening[] | null;
