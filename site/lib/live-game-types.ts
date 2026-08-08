export type LiveGamePhase = "starting" | "playing" | "paused" | "finished";
export type LiveGameSource = "arena" | "tournament";
export type LiveGameResult = "1-0" | "0-1" | "1/2-1/2";

export type LiveGame = {
  id: string;
  source: LiveGameSource;
  tournamentId: string | null;
  tournamentPairKey: string | null;
  tournamentGameIndex: number | null;
  whiteName: string;
  blackName: string;
  whiteModelReference: string | null;
  blackModelReference: string | null;
  openingName: string | null;
  phase: LiveGamePhase;
  status: string;
  moves: string[];
  lastMoveMs: number | null;
  result: LiveGameResult | null;
  revision: number;
  startedAt: number;
  updatedAt: number;
};

export type CompletedLiveGame = {
  id: string;
  whiteName: string;
  blackName: string;
  pgn: string;
  result: LiveGameResult;
  termination: string;
  recordedAt: number;
};

export type LiveGameResponse = {
  live: LiveGame | null;
  completed: CompletedLiveGame | null;
};

