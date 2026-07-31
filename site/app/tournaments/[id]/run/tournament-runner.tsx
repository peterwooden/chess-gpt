"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { loadBrowserModel, type BrowserChessModel } from "../../../arena/model";
import { playGame } from "../../../arena/play-game.mjs";
import type { GameOutcome } from "../../../arena/play-game";
import { formatDuration, formatScore } from "../../tournament-nav";
import {
  loadRunnerIdentity,
  describeMachine,
  type RunnerPlan,
  type ScheduledGame,
} from "./runner-identity.mjs";

const HEARTBEAT_INTERVAL_MS = 20_000;

type Phase = "idle" | "claiming" | "loading" | "playing" | "stopped" | "finished";

type Standing = {
  entryId: string;
  displayName: string;
  points: number;
  wins: number;
  draws: number;
  losses: number;
  games: number;
  rank: number;
};

export function TournamentRunner({ tournamentId }: { tournamentId: string }) {
  const [phase, setPhase] = useState<Phase>("idle");
  const [plan, setPlan] = useState<RunnerPlan | null>(null);
  const [status, setStatus] = useState("Not started.");
  const [error, setError] = useState<string | null>(null);
  const [label, setLabel] = useState("");
  const [override, setOverride] = useState(false);
  const [completed, setCompleted] = useState(0);
  const [remaining, setRemaining] = useState(0);
  const [standings, setStandings] = useState<Standing[]>([]);
  const [lastMoveMs, setLastMoveMs] = useState<number | null>(null);
  const [currentGame, setCurrentGame] = useState<string>("");
  const [ply, setPly] = useState(0);
  const [recentGames, setRecentGames] = useState<string[]>([]);
  const [startedAt, setStartedAt] = useState<number | null>(null);
  const [elapsedMs, setElapsedMs] = useState(0);
  // Mirrored into state because the machine identity is rendered, and a ref
  // read during render would not update the pinning notice when it changes.
  const [runnerId, setRunnerId] = useState<string | null>(null);

  const running = useRef(false);
  const models = useRef(new Map<string, BrowserChessModel>());
  const identity = useRef<ReturnType<typeof loadRunnerIdentity> | null>(null);

  // Deferred a tick because localStorage is unavailable during server render,
  // so the identity can only be read once the component is on the client.
  useEffect(() => {
    const timer = setTimeout(() => {
      const loaded = loadRunnerIdentity();
      identity.current = loaded;
      setRunnerId(loaded.id);
      if (loaded.label) setLabel(loaded.label);
    }, 0);
    return () => clearTimeout(timer);
  }, []);

  useEffect(() => {
    if (startedAt === null || (phase !== "playing" && phase !== "loading")) return;
    const timer = setInterval(() => setElapsedMs(currentTime() - startedAt), 1_000);
    return () => clearInterval(timer);
  }, [phase, startedAt]);

  // Release sessions if the operator navigates away mid-run.
  useEffect(() => () => {
    running.current = false;
    for (const model of models.current.values()) void model.dispose();
    models.current.clear();
  }, []);

  const refreshPlan = useCallback(async () => {
    const response = await fetch(`/api/tournaments/${tournamentId}/plan`);
    const payload = await response.json() as RunnerPlan & { error?: string };
    if (!response.ok) throw new Error(payload.error ?? "The tournament plan could not be loaded.");
    setPlan(payload);
    setCompleted(payload.playedCount);
    setRemaining(payload.remaining.length);
    return payload;
  }, [tournamentId]);

  useEffect(() => {
    let cancelled = false;
    const timer = setTimeout(() => {
      refreshPlan().catch((caught) => {
        if (cancelled) return;
        setError(caught instanceof Error ? caught.message : "The tournament plan could not be loaded.");
      });
    }, 0);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [refreshPlan]);

  // Keep the lease alive so a second tab cannot start and contend for the
  // processor, which would silently corrupt every move's time budget.
  useEffect(() => {
    if (!runnerId) return;
    if (phase !== "playing" && phase !== "loading") return;
    const timer = setInterval(() => {
      void fetch(`/api/tournaments/${tournamentId}/runner`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ runnerId, renew: true }),
      });
    }, HEARTBEAT_INTERVAL_MS);
    return () => clearInterval(timer);
  }, [phase, runnerId, tournamentId]);

  async function modelFor(entryId: string, current: RunnerPlan): Promise<BrowserChessModel> {
    const cached = models.current.get(entryId);
    // A package that forfeited on time had its worker terminated. It loses that
    // game only, so it is reloaded rather than dropped from the tournament.
    if (cached && cached.alive) return cached;
    if (cached) {
      models.current.delete(entryId);
      void cached.dispose();
    }
    const entry = current.entries.find((candidate) => candidate.id === entryId);
    if (!entry) throw new Error("A scheduled entry is missing from the tournament.");
    setStatus(`Loading ${entry.displayName}…`);
    const model = await loadBrowserModel(entry.reference, () => {});
    if (model.info.digest !== entry.manifestSha256) {
      await model.dispose();
      throw new Error(
        `${entry.displayName} no longer matches the manifest digest recorded at registration.`,
      );
    }
    models.current.set(entryId, model);
    return model;
  }

  async function start() {
    setError(null);
    if (!identity.current) return;
    setPhase("claiming");
    try {
      const response = await fetch(`/api/tournaments/${tournamentId}/runner`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          runnerId: identity.current?.id,
          runnerLabel: label,
          runnerMetadata: describeMachine(),
          override,
        }),
      });
      const payload = await response.json() as { error?: string };
      if (!response.ok) throw new Error(payload.error ?? "The runner lease could not be claimed.");
      identity.current?.saveLabel(label);
      setStartedAt(currentTime());
      running.current = true;
      setPhase("loading");
      void loop();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "The runner lease could not be claimed.");
      setPhase("idle");
    }
  }

  function stop() {
    running.current = false;
    setPhase("stopped");
    setStatus("Stopped. Press resume to continue from where it left off.");
  }

  async function loop() {
    try {
      let current = await refreshPlan();
      setPhase("playing");

      while (running.current) {
        const next: ScheduledGame | undefined = current.remaining[0];
        if (!next) {
          setPhase("finished");
          setStatus("All scheduled games are played.");
          return;
        }

        const white = current.entries.find((entry) => entry.id === next.whiteEntryId);
        const black = current.entries.find((entry) => entry.id === next.blackEntryId);
        if (!white || !black) throw new Error("A scheduled entry is missing from the tournament.");

        // Record the attempt before playing. A crash that takes the tab with it
        // still counts, so a package that reliably kills the runner eventually
        // exhausts its attempts instead of blocking the tournament forever.
        const attemptResponse = await fetch(`/api/tournaments/${tournamentId}/attempts`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ pairKey: next.pairKey, gameIndex: next.gameIndex }),
        });
        const attemptPayload = await attemptResponse.json() as { attempts?: number; error?: string };
        if (!attemptResponse.ok) {
          throw new Error(attemptPayload.error ?? "The attempt could not be recorded.");
        }
        if ((attemptPayload.attempts ?? 1) > current.tournament.maxAttemptsPerGame) {
          setStatus(`Abandoned ${white.displayName} v ${black.displayName} after repeated failures.`);
          current = await refreshPlan();
          continue;
        }

        setCurrentGame(`${white.displayName} (white) v ${black.displayName} (black)`);
        setPly(0);
        setStatus("Loading packages…");
        const [whiteModel, blackModel] = [
          await modelFor(next.whiteEntryId, current),
          await modelFor(next.blackEntryId, current),
        ];
        if (!running.current) return;

        setStatus("Playing…");
        const outcome: GameOutcome = await playGame(
          adapt(white.displayName, whiteModel),
          adapt(black.displayName, blackModel),
          {
            moveTimeLimitMs: current.tournament.moveTimeLimitMs,
            maxPlies: current.tournament.maxPlies,
            seed: seedFor(next),
            now: () => performance.now(),
            onMove: (move) => {
              setPly(move.ply);
              setLastMoveMs(Math.round(move.elapsedMs));
            },
          },
        );

        setStatus("Saving…");
        await save(next, white.reference, black.reference, outcome);

        setRecentGames((history) => [
          `${white.displayName} v ${black.displayName} — ${outcome.result} (${outcome.termination})`,
          ...history,
        ].slice(0, 12));

        current = await refreshPlan();
        await refreshStandings();
      }
    } catch (caught) {
      running.current = false;
      setPhase("stopped");
      setError(caught instanceof Error ? caught.message : "The tournament run stopped.");
    }
  }

  /**
   * Persist before starting the next game. Stopping on a persistent write
   * failure is deliberate: playing hundreds of games that are not being
   * recorded is worse than stopping and being told why.
   */
  async function save(
    slot: ScheduledGame,
    whiteReference: string,
    blackReference: string,
    outcome: GameOutcome,
  ) {
    const response = await fetch("/api/games", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        id: crypto.randomUUID(),
        pgn: outcome.pgn,
        playedAt: new Date(currentTime()).toISOString(),
        white: { kind: "model", reference: whiteReference },
        black: { kind: "model", reference: blackReference },
        tournament: {
          tournamentId,
          pairKey: slot.pairKey,
          gameIndex: slot.gameIndex,
        },
      }),
    });
    if (!response.ok) {
      const payload = await response.json() as { error?: string };
      throw new Error(payload.error ?? "The game could not be recorded.");
    }
  }

  async function refreshStandings() {
    const response = await fetch(`/api/tournaments/${tournamentId}/results`);
    if (!response.ok) return;
    const payload = await response.json() as { table?: Standing[] };
    if (payload.table) setStandings(payload.table);
  }

  const total = completed + remaining;
  const percent = total > 0 ? Math.round((completed / total) * 100) : 0;
  const eta = completed > 0 && elapsedMs > 0 && remaining > 0
    ? formatDuration((elapsedMs / completed) * remaining)
    : "—";

  return (
    <>
      <section className="tournament-panel">
        <h2>Machine</h2>
        <label className="tournament-runner-label">
          <span>This machine</span>
          <input
            value={label}
            onChange={(event) => setLabel(event.target.value)}
            placeholder="Sam's M2 Air"
            disabled={phase === "playing" || phase === "loading"}
          />
          <small>
            Recorded against the tournament. A tournament is pinned to the machine that
            starts it, because under a move clock a result split across machines is not a
            clean result.
          </small>
        </label>
        {plan?.tournament.runnerId && runnerId && plan.tournament.runnerId !== runnerId ? (
          <label className="tournament-override">
            <input
              type="checkbox"
              checked={override}
              onChange={(event) => setOverride(event.target.checked)}
            />
            <span>
              Move this tournament to this machine. Requires an administrator, and is
              recorded permanently beside the standings.
            </span>
          </label>
        ) : null}

        <div className="tournament-form-actions">
          {phase === "playing" || phase === "loading" ? (
            <button type="button" onClick={stop}>Stop</button>
          ) : (
            <button type="button" className="tournament-primary" onClick={start}>
              {completed > 0 ? "Resume" : "Start"}
            </button>
          )}
        </div>
        {error ? <p className="tournament-error">{error}</p> : null}
      </section>

      <section className="tournament-panel">
        <h2>Progress</h2>
        <div
          className="tournament-progress"
          role="progressbar"
          aria-valuenow={percent}
          aria-valuemin={0}
          aria-valuemax={100}
        >
          <div style={{ width: `${percent}%` }} />
        </div>
        <dl className="tournament-stats">
          <div><dt>Played</dt><dd>{completed} of {total}</dd></div>
          <div><dt>Status</dt><dd>{status}</dd></div>
          <div><dt>Current</dt><dd>{currentGame || "—"}</dd></div>
          <div><dt>Ply</dt><dd>{ply || "—"}</dd></div>
          <div><dt>Last move</dt><dd>{lastMoveMs === null ? "—" : `${lastMoveMs} ms`}</dd></div>
          <div><dt>Estimated remaining</dt><dd>{eta}</dd></div>
          {plan ? (
            <div><dt>Abandoned</dt><dd>{plan.abandonedCount}</dd></div>
          ) : null}
        </dl>
      </section>

      {standings.length > 0 ? (
        <section className="tournament-panel">
          <h2>Standings so far</h2>
          <table className="tournament-table">
            <thead>
              <tr>
                <th scope="col">#</th><th scope="col">Model</th><th scope="col">Points</th>
                <th scope="col">W</th><th scope="col">D</th><th scope="col">L</th>
              </tr>
            </thead>
            <tbody>
              {standings.map((row) => (
                <tr key={row.entryId}>
                  <td>{row.rank}</td>
                  <td>{row.displayName}</td>
                  <td><strong>{formatScore(row.points)}</strong></td>
                  <td>{row.wins}</td><td>{row.draws}</td><td>{row.losses}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      ) : null}

      {recentGames.length > 0 ? (
        <section className="tournament-panel">
          <h2>Recent results</h2>
          <ul className="tournament-recent">
            {recentGames.map((entry, index) => <li key={index}>{entry}</li>)}
          </ul>
        </section>
      ) : null}
    </>
  );
}

function adapt(name: string, model: BrowserChessModel) {
  return {
    name,
    newGame: (seed: number) => model.newGame(seed),
    predict: (history: string[], legalMoves: string[], limit: number) =>
      model.predict(history, legalMoves, limit),
  };
}

/** Isolated so the purity rule sees clock reads outside render. */
function currentTime(): number {
  return Date.now();
}

/** Derived from the slot so a replayed game gets the same seed it had before. */
function seedFor(slot: ScheduledGame): number {
  let hash = 2166136261;
  for (const character of `${slot.pairKey}#${slot.gameIndex}`) {
    hash ^= character.charCodeAt(0);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}
