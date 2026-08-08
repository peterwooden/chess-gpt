"use client";

import { Chess, type Color, type PieceSymbol } from "chess.js";
import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import type { LiveGameResponse } from "../../../lib/live-game-types";

const FILES = ["a", "b", "c", "d", "e", "f", "g", "h"] as const;
const RANKS = [8, 7, 6, 5, 4, 3, 2, 1] as const;
const PIECES: Record<Color, Record<PieceSymbol, string>> = {
  w: { p: "♟", n: "♞", b: "♝", r: "♜", q: "♛", k: "♚" },
  b: { p: "♟", n: "♞", b: "♝", r: "♜", q: "♛", k: "♚" },
};
const POLL_INTERVAL_MS = 2_000;
const STALE_AFTER_MS = 60_000;

export function LiveGameViewer({
  gameId,
  initial,
}: {
  gameId: string;
  initial: LiveGameResponse;
}) {
  const [response, setResponse] = useState(initial);
  const [now, setNow] = useState(() => Date.now());
  const [connectionError, setConnectionError] = useState(false);

  useEffect(() => {
    if (response.completed) return;
    let stopped = false;
    const refresh = async () => {
      try {
        const fetched = await fetch(`/api/live-games/${encodeURIComponent(gameId)}`, {
          cache: "no-store",
        });
        if (!fetched.ok) throw new Error("Live game unavailable.");
        const next = await fetched.json() as LiveGameResponse;
        if (!stopped) {
          setResponse(next);
          setConnectionError(false);
          setNow(Date.now());
        }
      } catch {
        if (!stopped) setConnectionError(true);
      }
    };
    const timer = window.setInterval(() => void refresh(), POLL_INTERVAL_MS);
    return () => {
      stopped = true;
      window.clearInterval(timer);
    };
  }, [gameId, response.completed]);

  const presentation = useMemo(() => present(response), [response]);
  const stale = Boolean(
    response.live
    && response.live.phase !== "finished"
    && now - response.live.updatedAt > STALE_AFTER_MS,
  );
  const lastMove = presentation.game.history({ verbose: true }).at(-1);
  const moveRows = pairMoves(presentation.moves);

  return (
    <main className="arena-page live-watch-page">
      <nav className="arena-nav" aria-label="Arena navigation">
        <Link href="/" className="arena-title">ChessGPT live</Link>
        <div className="arena-nav-links">
          <Link href="/arena">Arena</Link><Link href="/tournaments">Tournaments</Link>
        </div>
      </nav>

      <header className="live-watch-hero">
        <p className="eyebrow">{response.completed ? "Final position" : "Live game"}</p>
        <h1>{presentation.whiteName} <span>v</span> {presentation.blackName}</h1>
        <p aria-live="polite">
          {connectionError
            ? "Trying to reconnect…"
            : stale
              ? "The broadcaster has stopped updating. The game may be paused."
              : presentation.status}
        </p>
      </header>

      <section className="live-watch-layout" aria-label="Live chess game">
        <div className="live-board-stack">
          <PlayerBar color="b" name={presentation.blackName} />
          <div className="chessboard live-watch-board" role="grid" aria-label="Chess board">
            {RANKS.flatMap((rank) => FILES.map((file) => {
              const square = `${file}${rank}` as const;
              const piece = presentation.game.get(square);
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
                  {rank === 1 ? <small className="file-label">{file}</small> : null}
                  {file === "a" ? <small className="rank-label">{rank}</small> : null}
                </div>
              );
            }))}
          </div>
          <PlayerBar color="w" name={presentation.whiteName} />
        </div>

        <aside className="live-scorecard">
          <header>
            <div>
              <span>{presentation.openingName ?? "Game score"}</span>
              <strong>{presentation.moves.length} plies · {Math.ceil(presentation.moves.length / 2)} moves</strong>
            </div>
            <i className={!response.completed && !stale ? "pulse active" : "pulse"} aria-hidden="true" />
          </header>
          {moveRows.length === 0 ? (
            <div className="empty-record"><span>01</span><p>Waiting for the first move…</p></div>
          ) : (
            <div className="move-score">
              <div className="move-score-heading" aria-hidden="true">
                <span>Move</span><b>White</b><b>Black</b>
              </div>
              <ol className="move-record" aria-label="Move history">
                {moveRows.map((row) => (
                  <li key={row.number}>
                    <span>{row.number}.</span>
                    <strong>{row.white ?? "—"}</strong>
                    <strong>{row.black ?? "—"}</strong>
                  </li>
                ))}
              </ol>
            </div>
          )}
          <footer>
            {response.completed ? (
              <Link href={`/arena?game=${encodeURIComponent(gameId)}`}>Open the recorded game and review →</Link>
            ) : response.live?.tournamentId ? (
              <Link href={`/tournaments/${encodeURIComponent(response.live.tournamentId)}`}>Back to tournament standings →</Link>
            ) : (
              <span>Updates every two seconds</span>
            )}
          </footer>
        </aside>
      </section>
    </main>
  );
}

function present(response: LiveGameResponse) {
  if (response.completed) {
    const game = new Chess();
    game.loadPgn(response.completed.pgn);
    return {
      game,
      moves: game.history(),
      whiteName: response.completed.whiteName,
      blackName: response.completed.blackName,
      openingName: game.getHeaders().Opening ?? null,
      status: `${response.completed.result} · ${response.completed.termination}`,
    };
  }
  const live = response.live!;
  const game = new Chess();
  for (const san of live.moves) game.move(san);
  return {
    game,
    moves: live.moves,
    whiteName: live.whiteName,
    blackName: live.blackName,
    openingName: live.openingName,
    status: live.status,
  };
}

function pairMoves(moves: readonly string[]) {
  const rows: Array<{ number: number; white?: string; black?: string }> = [];
  moves.forEach((move, index) => {
    const rowIndex = Math.floor(index / 2);
    const row = rows[rowIndex] ?? { number: rowIndex + 1 };
    if (index % 2 === 0) row.white = move;
    else row.black = move;
    rows[rowIndex] = row;
  });
  return rows;
}

function PlayerBar({ color, name }: { color: Color; name: string }) {
  return (
    <section className={`player-strip ${color === "w" ? "white" : "black"}`}>
      <div className="player-identity">
        <span className="player-color">{color === "w" ? "White" : "Black"}</span>
        <strong>{name}</strong>
      </div>
    </section>
  );
}

