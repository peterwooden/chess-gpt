"use client";

import { Chess, type Color, type PieceSymbol } from "chess.js";
import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { LiveGameEventBatch, LiveGameEventPayload } from "../../../lib/live-game-events.mjs";
import type { LiveGame, LiveGameResponse } from "../../../lib/live-game-types";
import { GameProgressPanel, useGameTimeline } from "../../arena/game-progress-panel";
import { PlayerStrip, type PlayerClock } from "../../arena/player-strip";
import { ThinkingOverlay, useThinkingDisplay } from "../../arena/thinking-overlay";

const FILES = ["a", "b", "c", "d", "e", "f", "g", "h"] as const;
const RANKS = [8, 7, 6, 5, 4, 3, 2, 1] as const;
const PIECES: Record<Color, Record<PieceSymbol, string>> = {
  w: { p: "♟", n: "♞", b: "♝", r: "♜", q: "♛", k: "♚" },
  b: { p: "♟", n: "♞", b: "♝", r: "♜", q: "♛", k: "♚" },
};
const POLL_INTERVAL_MS = 500;
const STALE_AFTER_MS = 60_000;

export function LiveGameViewer({ gameId, initial }: { gameId: string; initial: LiveGameResponse }) {
  const [response, setResponse] = useState(initial);
  const [now, setNow] = useState(() => Date.now());
  const [connectionError, setConnectionError] = useState(false);
  const [showThinking, setShowThinking] = useState(false);
  const [orientation, setOrientation] = useState<Color>("w");
  const [clockAnchor, setClockAnchor] = useState(() => activeClockAnchor(initial.live));
  const cursor = useRef(initial.live?.eventSeq ?? 0);
  const replay = useRef(Promise.resolve());
  const livePosition = useRef(gameFromMoves(initial.live?.moves ?? []));
  const thinkingColor = useRef<Color | null>(
    initial.live ? gameFromMoves(initial.live.moves).turn() : null,
  );
  const {
    squares: thinkingSquares,
    arrows: thinkingArrows,
    sequence: thinkingSequence,
    apply: applyThinking,
    clear: clearThinking,
  } = useThinkingDisplay();
  const livePhase = response.live?.phase;

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
      setClockAnchor({ color: payload.color, startedAtMs: Date.now() });
      return;
    }
    if (payload.type === "move.played") {
      clearThinking();
      thinkingColor.current = null;
      setClockAnchor(null);
      return;
    }
    if (payload.type !== "game.updated") return;
    if (payload.update.phase === "finished") clearThinking();
    setClockAnchor(activeClockAnchor(payload.update));
    livePosition.current = gameFromMoves(payload.update.moves);
    setResponse((current) => current.live ? {
      ...current,
      live: {
        ...current.live,
        ...payload.update,
        moves: [...payload.update.moves],
        updatedAt: Date.now(),
      },
    } : current);
    setNow(Date.now());
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
    const timer = window.setInterval(() => setNow(Date.now()), 1_000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    if (!livePhase || livePhase === "finished") return;
    let stopped = false;
    let refreshing = false;
    const refresh = async () => {
      if (refreshing || stopped) return;
      refreshing = true;
      try {
        const fetched = await fetch(
          `/api/live-games/${encodeURIComponent(gameId)}?after=${cursor.current}`,
          { cache: "no-store" },
        );
        if (!fetched.ok) throw new Error("Live game unavailable.");
        const next = await fetched.json() as LiveGameResponse;
        if (stopped) return;
        setResponse((current) => ({ ...current, completed: next.completed ?? current.completed }));
        for (const batch of next.batches) consumeBatch(batch);
        if (next.live && next.live.eventSeq > cursor.current && next.batches.length === 0) {
          cursor.current = next.live.eventSeq;
          clearThinking();
          livePosition.current = gameFromMoves(next.live.moves);
          thinkingColor.current = next.live.phase === "playing"
            ? livePosition.current.turn()
            : null;
          setResponse(next);
          setClockAnchor(activeClockAnchor(next.live));
        }
        setConnectionError(false);
      } catch {
        if (!stopped) setConnectionError(true);
      } finally {
        refreshing = false;
      }
    };
    void refresh();
    const timer = window.setInterval(() => void refresh(), POLL_INTERVAL_MS);
    return () => {
      stopped = true;
      window.clearInterval(timer);
    };
  }, [clearThinking, consumeBatch, gameId, livePhase]);

  const presentation = useMemo(() => present(response), [response]);
  const progressMoves = presentation.game.history({ verbose: true }).map((move, index) => ({
    ply: index + 1,
    san: move.san,
    color: move.color,
  }));
  const gameIsOngoing = Boolean(response.live && response.live.phase !== "finished");
  const timeline = useGameTimeline(progressMoves.length, gameIsOngoing);
  const stale = Boolean(
    response.live
    && response.live.phase !== "finished"
    && now - response.live.updatedAt > STALE_AFTER_MS,
  );
  const finished = response.live?.phase === "finished" || (!response.live && Boolean(response.completed));
  const displayedGame = timeline.isLive
    ? presentation.game
    : gameFromMoves(presentation.moves.slice(0, timeline.displayPly));
  const lastMove = displayedGame.history({ verbose: true }).at(-1);
  const boardRanks = orientation === "w" ? RANKS : [...RANKS].reverse();
  const boardFiles = orientation === "w" ? FILES : [...FILES].reverse();
  const stripMoves = displayedGame.history({ verbose: true });
  const whiteSummary = {
    color: "w" as const,
    name: presentation.whiteName,
    modelReference: response.live?.whiteModelReference ?? null,
    moves: stripMoves,
    clock: timeline.isLive ? playerClock(response.live, clockAnchor, "w") : null,
  };
  const blackSummary = {
    color: "b" as const,
    name: presentation.blackName,
    modelReference: response.live?.blackModelReference ?? null,
    moves: stripMoves,
    clock: timeline.isLive ? playerClock(response.live, clockAnchor, "b") : null,
  };
  const topSummary = orientation === "w" ? blackSummary : whiteSummary;
  const bottomSummary = orientation === "w" ? whiteSummary : blackSummary;

  return (
    <main className="arena-page live-watch-page">
      <nav className="arena-nav" aria-label="Arena navigation">
        <Link href="/" className="arena-title">ChessGPT live</Link>
        <div className="arena-nav-links"><Link href="/arena">Arena</Link><Link href="/tournaments">Tournaments</Link></div>
      </nav>

      <header className="live-watch-hero">
        <p className="eyebrow">{finished ? "Final position" : "Live game"}</p>
        <h1>{presentation.whiteName} <span>v</span> {presentation.blackName}</h1>
        <p aria-live="polite">
          {connectionError ? "Trying to reconnect…" : stale ? "The broadcaster has stopped updating. The game may be paused." : presentation.status}
        </p>
      </header>

      <section className="live-watch-layout" aria-label="Live chess game">
        <div className="live-board-stack">
          <PlayerStrip {...topSummary} />
          <div className="thinking-board-wrap">
            <div className={`chessboard live-watch-board orientation-${orientation}`} role="grid" aria-label="Chess board">
              {boardRanks.flatMap((rank) => boardFiles.map((file) => {
                const square = `${file}${rank}` as const;
                const piece = displayedGame.get(square);
                const light = (FILES.indexOf(file) + rank) % 2 === 0;
                const last = square === lastMove?.from || square === lastMove?.to;
                return (
                  <div className={`board-square ${light ? "light" : "dark"}${last ? " last" : ""}`} role="gridcell" aria-label={`${square}${piece ? ` ${piece.color === "w" ? "white" : "black"} piece` : " empty"}`} key={square}>
                    {piece ? <span className={`piece ${piece.color}`}>{PIECES[piece.color][piece.type]}</span> : null}
                    {(orientation === "w" ? rank === 1 : rank === 8) ? <small className="file-label">{file}</small> : null}
                    {(orientation === "w" ? file === "a" : file === "h") ? <small className="rank-label">{rank}</small> : null}
                  </div>
                );
              }))}
            </div>
            <ThinkingOverlay enabled={showThinking && timeline.isLive} orientation={orientation} sequence={thinkingSequence} squares={thinkingSquares} arrows={thinkingArrows} />
          </div>
          <PlayerStrip {...bottomSummary} onFlip={() => setOrientation((color) => color === "w" ? "b" : "w")} />
        </div>

        <GameProgressPanel
          className="live-scorecard"
          label={presentation.openingName ?? "Game score"}
          liveStatus={presentation.status}
          moves={progressMoves}
          timeline={timeline}
          pulse={!finished && !stale}
          showThinking={showThinking}
          onShowThinkingChange={setShowThinking}
          emptyMessage="Waiting for the first move…"
        >
          <footer>
            {finished ? (
              <Link href={`/arena?game=${encodeURIComponent(gameId)}`}>Open the recorded game and review →</Link>
            ) : response.live?.tournamentId ? (
              <Link href={`/tournaments/${encodeURIComponent(response.live.tournamentId)}`}>Back to tournament standings →</Link>
            ) : (
              <span>Live updates · 500 ms polling</span>
            )}
          </footer>
        </GameProgressPanel>
      </section>
    </main>
  );
}

function present(response: LiveGameResponse) {
  if (response.live) {
    const game = gameFromMoves(response.live.moves);
    return { game, moves: response.live.moves, whiteName: response.live.whiteName, blackName: response.live.blackName, openingName: response.live.openingName, status: response.live.status };
  }
  const completed = response.completed!;
  const game = new Chess();
  game.loadPgn(completed.pgn);
  return { game, moves: game.history(), whiteName: completed.whiteName, blackName: completed.blackName, openingName: game.getHeaders().Opening ?? null, status: `${completed.result} · ${completed.termination}` };
}

function gameFromMoves(moves: readonly string[]): Chess {
  const game = new Chess();
  for (const san of moves) game.move(san);
  return game;
}

function activeClockAnchor(snapshot: {
  activeTurnColor?: Color | null;
  activeTurnElapsedMs?: number | null;
} | null | undefined): { color: Color; startedAtMs: number } | null {
  if (!snapshot?.activeTurnColor || snapshot.activeTurnElapsedMs === null
      || snapshot.activeTurnElapsedMs === undefined) return null;
  return {
    color: snapshot.activeTurnColor,
    startedAtMs: Date.now() - snapshot.activeTurnElapsedMs,
  };
}

function playerClock(
  live: LiveGame | null,
  anchor: { color: Color; startedAtMs: number } | null,
  color: Color,
): PlayerClock | null {
  if (!live || live.phase !== "playing" || anchor?.color !== color) return null;
  const modelReference = color === "w" ? live.whiteModelReference : live.blackModelReference;
  const limitMs = color === "w" ? live.whiteMoveTimeLimitMs : live.blackMoveTimeLimitMs;
  return modelReference && limitMs ? { startedAtMs: anchor.startedAtMs, limitMs } : null;
}

function wait(ms: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}
