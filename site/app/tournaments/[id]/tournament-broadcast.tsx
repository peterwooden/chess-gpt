"use client";

import { Chess, type Color, type PieceSymbol, type Square } from "chess.js";
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
const FILES = ["a", "b", "c", "d", "e", "f", "g", "h"] as const;
const RANKS = [8, 7, 6, 5, 4, 3, 2, 1] as const;
const PIECES: Record<Color, Record<PieceSymbol, string>> = {
  w: { p: "♟", n: "♞", b: "♝", r: "♜", q: "♛", k: "♚" },
  b: { p: "♟", n: "♞", b: "♝", r: "♜", q: "♛", k: "♚" },
};

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
    <section className="tournament-broadcast-dashboard" aria-labelledby="standings-title">
      <aside className="tournament-broadcast-rail">
        <header>
          <span>Live tournament</span>
          <h2 id="standings-title">Standings</h2>
          <p aria-live="polite">{broadcastStatus(state, delayed, stale)}</p>
        </header>

        <div className="tournament-progress-block">
          <div className="tournament-progress-copy">
            <span>Progress</span>
            <strong>{state.playedCount} / {state.scheduledCount}</strong>
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
          <small>{percent}% complete</small>
        </div>

        {state.status === "completed" && state.shared ? (
          <p className="tournament-shared-title">The leaders share the title.</p>
        ) : null}

        <ol className="tournament-ranking-list" aria-label="Ranked tournament standings">
          {state.table.map((row) => (
            <li key={row.entryId}>
              <span className="tournament-rank">{row.rank}</span>
              <div className="tournament-ranked-model">
                <strong>{row.displayName}</strong>
                <small>{row.games} games · {row.games > 0 ? `${Math.round((row.points / row.games) * 100)}%` : "—"}</small>
              </div>
              <strong className="tournament-points">{formatScore(row.points)}</strong>
              <div className="tournament-ranked-wdl" aria-label={`${row.wins} wins, ${row.draws} draws, ${row.losses} losses`}>
                <span><b>{row.wins}</b> W</span>
                <span><b>{row.draws}</b> D</span>
                <span><b>{row.losses}</b> L</span>
              </div>
            </li>
          ))}
        </ol>
      </aside>

      <div className="tournament-games-stage">
        <header>
          <div>
            <span>On the board</span>
            <h2>Current game</h2>
          </div>
          <small>Games run sequentially on the pinned tournament machine.</small>
        </header>
        <div className="tournament-game-grid">
          {state.liveGame ? (
            <TournamentLiveCard game={state.liveGame} stale={stale} />
          ) : (
            <div className="tournament-live-empty">
              <span aria-hidden="true">01</span>
              <div>
                <strong>{state.status === "completed" ? "Tournament complete" : "Waiting for the next pairing"}</strong>
                <p>{state.status === "completed" ? "Every scheduled game has finished." : "The runner will place the next live board here."}</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}

function TournamentLiveCard({ game, stale }: { game: LiveGame; stale: boolean }) {
  const position = gameFromMoves(game.moves);
  const lastMove = position.history({ verbose: true }).at(-1);
  const recentMoves = position.history().slice(-8);
  return (
    <article className="tournament-live-card">
      <header>
        <div>
          <span>{stale ? "Paused" : "Live now"}</span>
          <strong>{game.whiteName} <i>v</i> {game.blackName}</strong>
        </div>
        <b>{game.moves.length} plies</b>
      </header>
      <div className="tournament-live-card-body">
        <div className="chessboard tournament-live-board" role="grid" aria-label="Current tournament position">
          {RANKS.flatMap((rank) => FILES.map((file) => {
            const square = `${file}${rank}` as Square;
            const piece = position.get(square);
            const light = (FILES.indexOf(file) + rank) % 2 === 0;
            const last = square === lastMove?.from || square === lastMove?.to;
            return (
              <div
                className={`board-square ${light ? "light" : "dark"}${last ? " last" : ""}`}
                role="gridcell"
                aria-label={`${square}${piece ? ` ${piece.color === "w" ? "white" : "black"} piece` : " empty"}`}
                key={square}
              >
                {piece ? <span className={`piece ${piece.color}`}>{PIECES[piece.color][piece.type]}</span> : null}
              </div>
            );
          }))}
        </div>
        <div className="tournament-live-card-details">
          <div className="tournament-live-players">
            <span><i className="white" />{game.whiteName}</span>
            <span><i className="black" />{game.blackName}</span>
          </div>
          <dl>
            <div><dt>Status</dt><dd>{stale ? "Runner paused" : game.status}</dd></div>
            <div><dt>Opening</dt><dd>{game.openingName ?? "Standard position"}</dd></div>
            <div><dt>Last move</dt><dd>{lastMove?.san ?? "Waiting…"}</dd></div>
            <div><dt>Move time</dt><dd>{game.lastMoveMs === null ? "—" : `${Math.round(game.lastMoveMs)} ms`}</dd></div>
          </dl>
          <p className="tournament-recent-moves">{recentMoves.length > 0 ? recentMoves.join("  ") : "Waiting for the first move…"}</p>
          <Link className="tournament-watch-link" href={`/watch/${game.id}`}>
            Open live player + thinking →
          </Link>
        </div>
      </div>
    </article>
  );
}

function gameFromMoves(moves: readonly string[]): Chess {
  const game = new Chess();
  for (const san of moves) game.move(san);
  return game;
}

function broadcastStatus(state: BroadcastState, delayed: boolean, stale: boolean): string {
  if (delayed) return "Results are reconnecting…";
  if (state.liveGame) return stale ? "The current game appears paused." : `${state.liveGame.whiteName} v ${state.liveGame.blackName}`;
  if (state.status === "running") return "Waiting for the runner to begin the next game…";
  if (state.status === "completed") return "All scheduled games are complete.";
  return "Waiting for play to begin.";
}
