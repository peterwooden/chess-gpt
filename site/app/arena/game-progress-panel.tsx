"use client";

import { type Color } from "chess.js";
import { type ReactNode, useEffect, useRef, useState } from "react";

export type ProgressMoveReview = {
  judgement: "brilliant" | "good" | "inaccuracy" | "mistake" | "blunder";
  label: string;
  glyph: string;
  winningChanceLoss: number;
  bestMoveSan?: string | null;
};

export type ProgressMove = {
  ply: number;
  san: string;
  color: Color;
  actor?: string;
  source?: string;
  elapsedMs?: number | null;
  review?: ProgressMoveReview;
};

export type GameTimeline = {
  displayPly: number;
  isLive: boolean;
  gameIsOngoing: boolean;
  behind: number;
  playing: boolean;
  showPosition(ply: number | null): void;
  togglePlayback(): void;
  reset(): void;
};

export function useGameTimeline(moveCount: number, gameIsOngoing: boolean, onNavigate?: () => void): GameTimeline {
  const [viewedPly, setViewedPly] = useState<number | null>(null);
  const [playing, setPlaying] = useState(false);
  const displayPly = viewedPly === null ? moveCount : Math.min(viewedPly, moveCount);
  const isLive = viewedPly === null;

  useEffect(() => {
    if (!playing) return;
    if (moveCount === 0) return;
    const timer = window.setTimeout(() => {
      if (viewedPly === null || viewedPly >= moveCount - 1) {
        setViewedPly(null);
        setPlaying(false);
      } else {
        setViewedPly(viewedPly + 1);
      }
    }, 700);
    return () => window.clearTimeout(timer);
  }, [moveCount, playing, viewedPly]);

  function showPosition(ply: number | null) {
    onNavigate?.();
    setPlaying(false);
    setViewedPly(ply === null || ply >= moveCount ? null : Math.max(0, ply));
  }

  function togglePlayback() {
    onNavigate?.();
    if (gameIsOngoing) {
      setViewedPly(null);
      setPlaying(false);
      return;
    }
    if (playing) {
      setPlaying(false);
      return;
    }
    if (moveCount === 0) return;
    setViewedPly((current) => current === null ? 0 : current);
    setPlaying(true);
  }

  function reset() {
    setViewedPly(null);
    setPlaying(false);
  }

  return {
    displayPly,
    isLive,
    gameIsOngoing,
    behind: isLive ? 0 : moveCount - displayPly,
    playing,
    showPosition,
    togglePlayback,
    reset,
  };
}

export function GameProgressPanel({
  className = "",
  label,
  liveStatus,
  moves,
  timeline,
  pulse,
  showThinking,
  onShowThinkingChange,
  emptyMessage,
  openingOnlyHeader = false,
  children,
}: {
  className?: string;
  label: string | null;
  liveStatus: string;
  moves: readonly ProgressMove[];
  timeline: GameTimeline;
  pulse: boolean;
  showThinking: boolean;
  onShowThinkingChange(enabled: boolean): void;
  emptyMessage: string;
  openingOnlyHeader?: boolean;
  children?: ReactNode;
}) {
  const moveRecordRef = useRef<HTMLOListElement>(null);
  const rows = pairMoves(moves);
  const selectedMove = timeline.displayPly > 0 ? moves[timeline.displayPly - 1] : undefined;
  const status = timeline.isLive
    ? liveStatus
    : timeline.displayPly === 0
      ? "Starting position"
      : `After ${selectedMove?.color === "w" ? "White" : "Black"} played ${selectedMove?.san ?? ""}`;

  useEffect(() => {
    const record = moveRecordRef.current;
    if (!record) return;
    if (timeline.isLive) {
      record.scrollTop = record.scrollHeight;
      return;
    }
    if (timeline.displayPly === 0) {
      record.scrollTop = 0;
      return;
    }
    const selected = record.querySelector<HTMLElement>(`[data-ply="${timeline.displayPly}"]`);
    const row = selected?.closest("li");
    if (!row) return;
    const rowBottom = row.offsetTop + row.offsetHeight;
    if (row.offsetTop < record.scrollTop) record.scrollTop = row.offsetTop;
    else if (rowBottom > record.scrollTop + record.clientHeight) {
      record.scrollTop = rowBottom - record.clientHeight;
    }
  }, [moves.length, timeline.displayPly, timeline.isLive]);

  return (
    <section className={`move-console game-progress-panel${className ? ` ${className}` : ""}`} aria-label="Game progress">
      {openingOnlyHeader ? (
        label ? <header className="opening-only-progress-header"><strong>{label}</strong></header> : null
      ) : (
        <header>
          <div>
            <span>{label}</span>
            <strong aria-live="polite">{status}</strong>
          </div>
          <div className="move-header-meta">
            {timeline.isLive ? (
              <small>{moves.length} plies · {Math.ceil(moves.length / 2)} moves</small>
            ) : (
              <small className="history-view-label">
                History{timeline.behind > 0 ? ` · ${timeline.behind} ${timeline.behind === 1 ? "ply" : "plies"} behind` : ""}
              </small>
            )}
            <i className={pulse && timeline.isLive ? "pulse active" : "pulse"} aria-hidden="true" />
          </div>
        </header>
      )}
      <label className="thinking-display-option">
        <input
          type="checkbox"
          checked={showThinking}
          onChange={(event) => onShowThinkingChange(event.target.checked)}
        />
        <span>Show model thinking</span>
      </label>
      <HistoryNavigation timeline={timeline} moveCount={moves.length} />
      {moves.length === 0 ? (
        <div className="empty-record"><span>01</span><p>{emptyMessage}</p></div>
      ) : (
        <div className="move-score">
          <div className="move-score-heading" aria-hidden="true">
            <span>Move</span><b>White</b><b>Black</b>
          </div>
          <ol className="move-record" aria-label="Move history" ref={moveRecordRef}>
            {rows.map((row) => (
              <li key={row.number}>
                <span>{row.number}.</span>
                <MoveCell move={row.white} timeline={timeline} />
                <MoveCell move={row.black} timeline={timeline} />
              </li>
            ))}
          </ol>
        </div>
      )}
      {children}
    </section>
  );
}

function HistoryNavigation({ timeline, moveCount }: { timeline: GameTimeline; moveCount: number }) {
  return (
    <nav className="history-navigation" aria-label="Move history navigation">
      <button type="button" aria-label="Go to starting position" onClick={() => timeline.showPosition(0)} disabled={moveCount === 0 || timeline.displayPly === 0}>
        <span aria-hidden="true">|‹</span>
      </button>
      <button type="button" aria-label="Go to previous position" onClick={() => timeline.showPosition(timeline.displayPly - 1)} disabled={moveCount === 0 || timeline.displayPly === 0}>
        <span aria-hidden="true">‹</span>
      </button>
      <button className="history-playback" type="button" aria-label={timeline.gameIsOngoing ? "Go to live position" : timeline.playing ? "Pause move history" : "Play move history to live"} aria-pressed={timeline.playing} onClick={timeline.togglePlayback} disabled={moveCount === 0}>
        <span aria-hidden="true">{timeline.playing ? "Ⅱ" : "▶"}</span>
      </button>
      <button type="button" aria-label="Go to next position" onClick={() => timeline.showPosition(timeline.displayPly + 1)} disabled={timeline.isLive || moveCount === 0}>
        <span aria-hidden="true">›</span>
      </button>
      <button type="button" aria-label="Go to live position" onClick={() => timeline.showPosition(null)} disabled={timeline.isLive || moveCount === 0}>
        <span aria-hidden="true">›|</span>
      </button>
    </nav>
  );
}

function MoveCell({ move, timeline }: { move?: ProgressMove; timeline: GameTimeline }) {
  if (!move) return <span className="score-move empty" aria-hidden="true">—</span>;
  const reviewDetail = move.review
    ? ` · ${move.review.label} · ${Math.round(move.review.winningChanceLoss)}% winning chance lost${move.review.bestMoveSan ? ` · best was ${move.review.bestMoveSan}` : ""}`
    : "";
  const moveDetail = [move.actor, move.source, move.elapsedMs === null || move.elapsedMs === undefined ? null : `${Math.round(move.elapsedMs)} ms`]
    .filter(Boolean)
    .join(" · ");
  const detail = `${moveDetail}${reviewDetail}`.replace(/^ · /, "");
  return (
    <button
      type="button"
      className={`score-move${move.review ? ` ${move.review.judgement}` : ""}${move.ply === timeline.displayPly ? " current" : ""}`}
      aria-label={`${move.color === "w" ? "White" : "Black"} played ${move.san}${detail ? `. ${detail}` : ""}`}
      aria-pressed={move.ply === timeline.displayPly}
      data-ply={move.ply}
      onClick={() => timeline.showPosition(move.ply)}
      title={detail || undefined}
    >
      <strong>{move.san}</strong>
      {move.review ? (
        <>
          <span className={`review-pill ${move.review.judgement} active`} title={move.review.label}>{move.review.glyph}</span>
          {move.review.winningChanceLoss >= 0.5 ? <small className="review-loss">−{Math.round(move.review.winningChanceLoss)}%</small> : null}
        </>
      ) : null}
    </button>
  );
}

function pairMoves(moves: readonly ProgressMove[]) {
  const rows: Array<{ number: number; white?: ProgressMove; black?: ProgressMove }> = [];
  for (const move of moves) {
    const rowIndex = Math.floor((move.ply - 1) / 2);
    const row = rows[rowIndex] ?? { number: rowIndex + 1 };
    if (move.color === "w") row.white = move;
    else row.black = move;
    rows[rowIndex] = row;
  }
  return rows;
}
