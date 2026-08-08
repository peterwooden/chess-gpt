"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import type { LiveGame } from "../../../lib/live-game-types";
import { formatScore } from "../tournament-nav";

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

type BroadcastState = {
  table: Standing[];
  shared: boolean;
  playedCount: number;
  scheduledCount: number;
  status: "registration" | "running" | "completed";
  liveGame: LiveGame | null;
};

const POLL_INTERVAL_MS = 2_000;
const STALE_AFTER_MS = 60_000;

export function TournamentBroadcast({
  tournamentId,
  initial,
}: {
  tournamentId: string;
  initial: BroadcastState;
}) {
  const [state, setState] = useState(initial);
  const [now, setNow] = useState(() => Date.now());
  const [delayed, setDelayed] = useState(false);

  useEffect(() => {
    if (state.status === "completed") return;
    let stopped = false;
    const refresh = async () => {
      try {
        const response = await fetch(`/api/tournaments/${tournamentId}/results`, { cache: "no-store" });
        if (!response.ok) throw new Error("Results unavailable.");
        const next = await response.json() as BroadcastState;
        if (!stopped) {
          setState(next);
          setNow(Date.now());
          setDelayed(false);
        }
      } catch {
        if (!stopped) setDelayed(true);
      }
    };
    const timer = window.setInterval(() => void refresh(), POLL_INTERVAL_MS);
    return () => {
      stopped = true;
      window.clearInterval(timer);
    };
  }, [state.status, tournamentId]);

  const percent = state.scheduledCount > 0
    ? Math.round((state.playedCount / state.scheduledCount) * 100)
    : 0;
  const stale = Boolean(state.liveGame && now - state.liveGame.updatedAt > STALE_AFTER_MS);

  return (
    <section className="tournament-panel" aria-labelledby="standings-title">
      <div className="tournament-live-heading">
        <div>
          <h2 id="standings-title">Tournament broadcast</h2>
          <p className="tournament-note" aria-live="polite">
            {delayed
              ? "Results are reconnecting…"
              : state.liveGame
                ? stale
                  ? "The current game appears paused."
                  : `${state.liveGame.whiteName} v ${state.liveGame.blackName} · ply ${state.liveGame.moves.length}`
                : state.status === "running"
                  ? "Waiting for the runner to begin the next game…"
                  : state.status === "completed" ? "All scheduled games are complete." : "Waiting for play to begin."}
          </p>
        </div>
        {state.liveGame ? (
          <Link className="tournament-watch-link" href={`/watch/${state.liveGame.id}`}>
            Watch current game →
          </Link>
        ) : null}
      </div>

      <div
        className="tournament-progress"
        role="progressbar"
        aria-valuenow={percent}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={`${state.playedCount} of ${state.scheduledCount} games complete`}
      >
        <div style={{ width: `${percent}%` }} />
      </div>
      <p className="tournament-progress-copy">
        <strong>{state.playedCount}</strong> of {state.scheduledCount} games · {percent}% complete
      </p>

      {state.status === "completed" && state.shared ? (
        <p className="tournament-note">
          The leaders are level on points and share the title. No further games are played to separate them.
        </p>
      ) : null}

      {state.table.length > 0 ? (
        <table className="tournament-table">
          <thead>
            <tr>
              <th scope="col">#</th><th scope="col">Model</th><th scope="col">Points</th>
              <th scope="col">W</th><th scope="col">D</th><th scope="col">L</th>
              <th scope="col">Games</th><th scope="col">Score</th>
            </tr>
          </thead>
          <tbody>
            {state.table.map((row) => (
              <tr key={row.entryId}>
                <td>{row.rank}</td>
                <td>{row.displayName}</td>
                <td><strong>{formatScore(row.points)}</strong></td>
                <td>{row.wins}</td><td>{row.draws}</td><td>{row.losses}</td><td>{row.games}</td>
                <td>{row.games > 0 ? `${Math.round((row.points / row.games) * 100)}%` : "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : null}
    </section>
  );
}

