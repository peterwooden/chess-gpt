export type RunnerIdentity = {
  id: string;
  label: string;
  saveLabel(label: string): void;
};

export type ScheduledGame = {
  pairKey: string;
  gameIndex: number;
  whiteEntryId: string;
  blackEntryId: string;
};

export type PlanEntry = {
  id: string;
  displayName: string;
  reference: string;
  manifestSha256: string;
  packageBytes: number;
};

export type RunnerPlan = {
  tournament: {
    id: string;
    name: string;
    status: "registration" | "running" | "completed";
    gamesPerPair: number;
    moveTimeLimitMs: number;
    maxPlies: number;
    maxAttemptsPerGame: number;
    runnerId: string | null;
    runnerLabel: string | null;
  };
  entries: PlanEntry[];
  scheduledCount: number;
  playedCount: number;
  abandonedCount: number;
  remaining: ScheduledGame[];
};

export function loadRunnerIdentity(): RunnerIdentity;
export function describeMachine(): Record<string, unknown>;
