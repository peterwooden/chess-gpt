"use client";

import { Chess, type Color, type PieceSymbol, type Square } from "chess.js";
import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import type { LiveGameEventBatch, LiveGameEventPayload } from "../../../lib/live-game-events.mjs";
import type { LiveGame } from "../../../lib/live-game-types";
import { ThinkingOverlay, useThinkingDisplay } from "../../arena/thinking-overlay";
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

const POLL_INTERVAL_MS = 500;
const GAME_POLL_INTERVAL_MS = 500;
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
  const [showThinking, setShowThinking] = useState(false);

  useEffect(() => {
    if (state.status === "completed") return;
    let stopped = false;
    const refresh = async () => {
      try {
        const response = await fetch(`/api/tournaments/${tournamentId}/results`, { cache: "no-store" });
        if (!response.ok) throw new Error("Results unavailable.");
        const next = await response.json() as BroadcastState;
        if (!stopped) {
          if (next.status !== state.status) {
            window.location.reload();
            return;
          }
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
  const stale = Boolean(
    state.liveGame
    && state.liveGame.phase !== "finished"
    && now - state.liveGame.updatedAt > STALE_AFTER_MS,
  );

  if (state.status === "registration") return null;

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
            <TournamentLiveCard
              key={state.liveGame.id}
              game={state.liveGame}
              stale={stale}
              showThinking={showThinking}
              onShowThinkingChange={setShowThinking}
            />
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

function TournamentLiveCard({
  game,
  stale,
  showThinking,
  onShowThinkingChange,
}: {
  game: LiveGame;
  stale: boolean;
  showThinking: boolean;
  onShowThinkingChange(show: boolean): void;
}) {
  const [live, setLive] = useState(game);
  const [connectionError, setConnectionError] = useState(false);
  const cursor = useRef(game.eventSeq);
  const replay = useRef(Promise.resolve());
  const livePosition = useRef(gameFromMoves(game.moves));
  const thinkingColor = useRef<Color | null>(
    game.phase === "playing" ? gameFromMoves(game.moves).turn() : null,
  );
  const {
    squares: thinkingSquares,
    arrows: thinkingArrows,
    sequence: thinkingSequence,
    apply: applyThinking,
    clear: clearThinking,
  } = useThinkingDisplay();

  const applyEvent = useCallback((payload: LiveGameEventPayload) => {
    if (payload.type === "thinking.command") {
      applyThinking(payload.command, {
        thinkingColor: thinkingColor.current,
        sourceColor: payload.command.type === "drawArrow"
          ? livePosition.current.get(payload.command.from)?.color ?? null
          : null,
      });
      return;
    }
    if (payload.type === "turn.started") {
      clearThinking();
      thinkingColor.current = payload.color;
      return;
    }
    if (payload.type === "move.played") {
      clearThinking();
      thinkingColor.current = null;
      return;
    }
    if (payload.type !== "game.updated") return;
    if (payload.update.phase === "finished") clearThinking();
    livePosition.current = gameFromMoves(payload.update.moves);
    setLive((current) => ({
      ...current,
      ...payload.update,
      moves: [...payload.update.moves],
      updatedAt: Date.now(),
    }));
  }, [applyThinking, clearThinking]);

  const consumeBatch = useCallback((batch: LiveGameEventBatch) => {
    if (batch.lastSeq <= cursor.current) return;
    cursor.current = batch.lastSeq;
    replay.current = replay.current.then(async () => {
      let previousOffset = batch.events[0]?.offsetMs ?? 0;
      for (const event of batch.events) {
        const delay = Math.min(500, Math.max(0, event.offsetMs - previousOffset));
        if (delay > 0) await wait(delay);
        applyEvent(event.payload);
        previousOffset = event.offsetMs;
      }
    });
  }, [applyEvent]);

  useEffect(() => {
    if (live.phase === "finished") return;
    let stopped = false;
    let refreshing = false;
    const refresh = async () => {
      if (refreshing || stopped) return;
      refreshing = true;
      try {
        const response = await fetch(
          `/api/live-games/${encodeURIComponent(game.id)}?after=${cursor.current}`,
          { cache: "no-store" },
        );
        if (!response.ok) throw new Error("Live game unavailable.");
        const next = await response.json() as { live: LiveGame | null; batches: LiveGameEventBatch[] };
        if (stopped) return;
        for (const batch of next.batches) consumeBatch(batch);
        if (next.live && next.live.eventSeq > cursor.current && next.batches.length === 0) {
          cursor.current = next.live.eventSeq;
          clearThinking();
          livePosition.current = gameFromMoves(next.live.moves);
          thinkingColor.current = next.live.phase === "playing"
            ? livePosition.current.turn()
            : null;
          setLive(next.live);
        }
        setConnectionError(false);
      } catch {
        if (!stopped) setConnectionError(true);
      } finally {
        refreshing = false;
      }
    };
    void refresh();
    const timer = window.setInterval(() => void refresh(), GAME_POLL_INTERVAL_MS);
    return () => {
      stopped = true;
      window.clearInterval(timer);
    };
  }, [clearThinking, consumeBatch, game.id, live.phase]);

  const position = gameFromMoves(live.moves);
  const lastMove = position.history({ verbose: true }).at(-1);
  const recentMoves = position.history().slice(-8);
  return (
    <article className="tournament-live-card">
      <header>
        <div>
          <span>{connectionError ? "Reconnecting" : live.phase === "finished" ? "Final" : stale ? "Paused" : "Live now"}</span>
          <strong>{live.whiteName} <i>v</i> {live.blackName}</strong>
        </div>
        <b>{live.moves.length} plies</b>
      </header>
      <div className="tournament-live-card-body">
        <div className="thinking-board-wrap tournament-live-board-wrap">
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
          <ThinkingOverlay enabled={showThinking} orientation="w" sequence={thinkingSequence} squares={thinkingSquares} arrows={thinkingArrows} />
        </div>
        <div className="tournament-live-card-details">
          <div className="tournament-live-players">
            <span><i className="white" />{live.whiteName}</span>
            <span><i className="black" />{live.blackName}</span>
          </div>
          <label className="thinking-display-option tournament-thinking-option">
            <input
              type="checkbox"
              checked={showThinking}
              onChange={(event) => onShowThinkingChange(event.target.checked)}
            />
            <span>Show model thinking</span>
          </label>
          <dl>
            <div><dt>Status</dt><dd>{connectionError ? "Reconnecting" : stale ? "Runner paused" : live.status}</dd></div>
            <div><dt>Opening</dt><dd>{live.openingName ?? "Standard position"}</dd></div>
            <div><dt>Last move</dt><dd>{lastMove?.san ?? "Waiting…"}</dd></div>
            <div><dt>Move time</dt><dd>{live.lastMoveMs === null ? "—" : `${Math.round(live.lastMoveMs)} ms`}</dd></div>
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
  if (state.status === "completed") return "All scheduled games are complete.";
  if (state.liveGame?.phase === "finished") return "Preparing the next pairing…";
  if (state.liveGame) return stale ? "The current game appears paused." : `${state.liveGame.whiteName} v ${state.liveGame.blackName}`;
  if (state.status === "running") return "Waiting for the runner to begin the next game…";
  return "Waiting for play to begin.";
}

function wait(ms: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}
