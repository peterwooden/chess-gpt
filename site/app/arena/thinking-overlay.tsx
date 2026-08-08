"use client";

import type { CSSProperties } from "react";
import { useCallback, useEffect, useRef, useState } from "react";
import type { Color, Square } from "chess.js";
import type { ThinkingCommand } from "../../lib/thinking-events.mjs";

type Mark = { key: string; intensity: number; fadeMs: number };
type SquareMark = Mark & { square: Square };
type ArrowMark = Mark & { from: Square; to: Square };

export function useThinkingDisplay() {
  const [squares, setSquares] = useState<Record<string, SquareMark>>({});
  const [arrows, setArrows] = useState<Record<string, ArrowMark>>({});
  const timers = useRef(new Set<number>());
  const revision = useRef(0);

  const clear = useCallback(() => {
    revision.current += 1;
    for (const timer of timers.current) window.clearTimeout(timer);
    timers.current.clear();
    setSquares({});
    setArrows({});
  }, []);

  const apply = useCallback((command: ThinkingCommand) => {
    if (command.type === "clearAll") {
      clear();
      return;
    }
    if (command.type === "clearSquare") {
      setSquares((current) => withoutKey(current, command.square));
      return;
    }
    if (command.type === "clearArrow") {
      setArrows((current) => withoutKey(current, arrowKey(command.from, command.to)));
      return;
    }

    const key = command.type === "highlightSquare"
      ? command.square
      : arrowKey(command.from, command.to);
    const mark = { ...command, key: `${key}-${++revision.current}` };
    if (command.type === "highlightSquare") {
      setSquares((current) => ({ ...current, [key]: mark }));
    } else {
      setArrows((current) => ({ ...current, [key]: mark }));
    }
    const timer = window.setTimeout(() => {
      timers.current.delete(timer);
      if (command.type === "highlightSquare") {
        setSquares((current) => current[key]?.key === mark.key ? withoutKey(current, key) : current);
      } else {
        setArrows((current) => current[key]?.key === mark.key ? withoutKey(current, key) : current);
      }
    }, Math.min(command.fadeMs, 2_147_483_647));
    timers.current.add(timer);
  }, [clear]);

  useEffect(() => clear, [clear]);
  return { squares, arrows, apply, clear };
}

export function ThinkingOverlay({
  enabled,
  orientation,
  squares,
  arrows,
}: {
  enabled: boolean;
  orientation: Color;
  squares: Record<string, SquareMark>;
  arrows: Record<string, ArrowMark>;
}) {
  if (!enabled) return null;
  return (
    <div className="thinking-overlay" aria-hidden="true">
      {Object.values(squares).map((mark) => {
        const point = squareCenter(mark.square, orientation);
        return <span className="thinking-square" key={mark.key} style={markStyle(mark, point)} />;
      })}
      {Object.values(arrows).map((mark) => {
        const from = squareCenter(mark.from, orientation);
        const to = squareCenter(mark.to, orientation);
        const dx = to.x - from.x;
        const dy = to.y - from.y;
        const length = Math.hypot(dx, dy);
        const style = {
          ...markStyle(mark, from),
          width: `${length}%`,
          transform: `translateY(-50%) rotate(${Math.atan2(dy, dx)}rad)`,
        };
        return <span className="thinking-arrow" key={mark.key} style={style} />;
      })}
    </div>
  );
}

function arrowKey(from: Square, to: Square): string {
  return `${from}-${to}`;
}

function withoutKey<T>(record: Record<string, T>, key: string): Record<string, T> {
  if (!(key in record)) return record;
  const next = { ...record };
  delete next[key];
  return next;
}

function squareCenter(square: Square, orientation: Color) {
  const file = square.charCodeAt(0) - 97;
  const rank = Number(square[1]) - 1;
  return orientation === "w"
    ? { x: (file + 0.5) * 12.5, y: (7.5 - rank) * 12.5 }
    : { x: (7.5 - file) * 12.5, y: (rank + 0.5) * 12.5 };
}

function markStyle(mark: Mark, point: { x: number; y: number }): CSSProperties {
  return {
    left: `${point.x}%`,
    top: `${point.y}%`,
    "--thinking-opacity": Math.min(0.5, mark.intensity * 0.5),
    "--thinking-fade-ms": `${mark.fadeMs}ms`,
  } as CSSProperties;
}
