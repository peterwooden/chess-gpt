"use client";

import { type Color, type PieceSymbol } from "chess.js";
import Link from "next/link";
import { type CSSProperties, useEffect, useState } from "react";
import { modelPageHref } from "./hugging-face-reference.mjs";

const PIECES: Record<Color, Record<PieceSymbol, string>> = {
  w: { p: "♟", n: "♞", b: "♝", r: "♜", q: "♛", k: "♚" },
  b: { p: "♟", n: "♞", b: "♝", r: "♜", q: "♛", k: "♚" },
};
const CAPTURE_VALUES: Record<PieceSymbol, number> = { p: 1, n: 3, b: 3, r: 5, q: 9, k: 0 };
const CAPTURE_ORDER: PieceSymbol[] = ["q", "r", "b", "n", "p"];

export type PlayerStripMove = {
  color: Color;
  captured?: PieceSymbol;
};

export type PlayerClock = {
  startedAtMs: number | null;
  limitMs: number;
};

export function PlayerStrip({
  color,
  name,
  moves,
  profileId,
  modelReference,
  clock,
  onFlip,
}: {
  color: Color;
  name: string;
  moves: readonly PlayerStripMove[];
  profileId?: string | null;
  modelReference?: string | null;
  clock?: PlayerClock | null;
  onFlip?: () => void;
}) {
  const colorName = color === "w" ? "White" : "Black";
  const capturedColor = oppositeColor(color);
  const captured = capturedPieces(moves, color);
  const ownPoints = capturePoints(captured);
  const opponentPoints = capturePoints(capturedPieces(moves, capturedColor));
  const lead = Math.max(0, ownPoints - opponentPoints);

  return (
    <section className={`player-strip ${color === "w" ? "white" : "black"}`} aria-label={`${colorName} player`}>
      <div className="player-heading">
        <div className="player-identity">
          <span className="player-color">{colorName}</span>
          {modelReference
            ? <ModelNameWithCopy name={name} reference={modelReference} />
            : profileId ? <Link href={`/players/${profileId}`}>{name}</Link> : <strong>{name}</strong>}
        </div>
        {clock ? <MoveClock clock={clock} /> : null}
      </div>
      <div className="player-material-row">
        <div className="captured-pieces" aria-label={`${colorName} captured pieces`}>
          {captured.length > 0 ? captured.map((piece, index) => (
            <span
              className={`captured-piece ${capturedColor === "w" ? "white" : "black"}`}
              aria-label={`Captured ${capturedColor === "w" ? "white" : "black"} ${pieceName(piece)}`}
              key={`${piece}-${index}`}
            >
              {PIECES[capturedColor][piece]}
            </span>
          )) : <span className="no-captures" aria-hidden="true">—</span>}
          {lead > 0 ? <strong className="material-lead">+{lead}</strong> : null}
        </div>
        {onFlip ? (
          <button className="board-flip-button" type="button" onClick={onFlip} aria-label="Flip board" title="Flip board">
            <span aria-hidden="true">⇅</span>
          </button>
        ) : null}
      </div>
    </section>
  );
}

function MoveClock({ clock }: { clock: PlayerClock }) {
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    if (clock.startedAtMs === null) return;
    const timer = window.setInterval(() => setNow(Date.now()), 100);
    return () => window.clearInterval(timer);
  }, [clock.startedAtMs]);

  const remainingMs = clock.startedAtMs === null ? clock.limitMs : Math.max(0, clock.limitMs - (now - clock.startedAtMs));
  const progress = clock.limitMs > 0 ? remainingMs / clock.limitMs : 0;
  const label = remainingMs < 10_000
    ? (remainingMs / 1_000).toFixed(1)
    : String(Math.ceil(remainingMs / 1_000));
  const style = { "--clock-progress": progress } as CSSProperties;

  return (
    <span
      className="move-clock"
      role="timer"
      aria-label={`${Math.ceil(remainingMs)} milliseconds remaining`}
      title={`${Math.ceil(remainingMs)} ms remaining`}
      style={style}
    >
      <span>{label}</span>
    </span>
  );
}

export function ModelNameWithCopy({ name, reference }: { name: string; reference: string }) {
  const [copied, setCopied] = useState(false);
  const [revealed, setRevealed] = useState(false);

  const copyReference = async () => {
    try {
      await navigator.clipboard.writeText(reference);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1800);
    } catch {
      setCopied(false);
    }
  };

  return (
    <span className={`model-name-with-copy${copied ? " copied" : ""}${revealed ? " revealed" : ""}`}>
      <Link
        href={modelPageHref(reference)}
        title={name}
        onClick={(event) => {
          if (!revealed && window.matchMedia("(hover: none)").matches) {
            event.preventDefault();
            setRevealed(true);
          }
        }}
      >
        {name}
      </Link>
      <button
        className="model-name-copy"
        type="button"
        aria-label={`Copy full reference for ${name}`}
        title={copied ? "Reference copied" : "Copy full reference"}
        onClick={() => void copyReference()}
      >
        <span aria-hidden="true">{copied ? "✓" : "⧉"}</span>
      </button>
    </span>
  );
}

function capturedPieces(moves: readonly PlayerStripMove[], color: Color): PieceSymbol[] {
  return moves
    .flatMap((move) => move.color === color && move.captured ? [move.captured] : [])
    .sort((left, right) => CAPTURE_ORDER.indexOf(left) - CAPTURE_ORDER.indexOf(right));
}

function capturePoints(pieces: readonly PieceSymbol[]): number {
  return pieces.reduce((total, piece) => total + CAPTURE_VALUES[piece], 0);
}

function pieceName(piece: PieceSymbol): string {
  return { p: "pawn", n: "knight", b: "bishop", r: "rook", q: "queen", k: "king" }[piece];
}

function oppositeColor(color: Color): Color {
  return color === "w" ? "b" : "w";
}
