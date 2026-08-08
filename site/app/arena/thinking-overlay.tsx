"use client";

import { useCallback, useId, useState } from "react";
import type { Color, Square } from "chess.js";
import type { ThinkingCommand } from "../../lib/thinking-events.mjs";
import {
  annotationOpacity,
  arrowSide,
  squareCenter,
  thinkingArrowPoints,
  type ArrowSide,
} from "../../lib/thinking-visuals.mjs";

const MAX_NEWER_ANNOTATIONS = 20;
const OWN_ARROW_COLOR = "#2f9e65";
const OPPONENT_ARROW_COLOR = "#d95c4f";

type Mark = { key: string; intensity: number; sequence: number };
type SquareMark = Mark & { square: Square };
type ArrowMark = Mark & { from: Square; to: Square; side: ArrowSide };
type ThinkingContext = { thinkingColor?: Color | null; sourceColor?: Color | null };
type DisplayState = {
  sequence: number;
  squares: Record<string, SquareMark>;
  arrows: Record<string, ArrowMark>;
};

const EMPTY_DISPLAY: DisplayState = { sequence: 0, squares: {}, arrows: {} };

export function useThinkingDisplay() {
  const [display, setDisplay] = useState<DisplayState>(EMPTY_DISPLAY);

  const clear = useCallback(() => {
    setDisplay((current) => ({ ...current, squares: {}, arrows: {} }));
  }, []);

  const apply = useCallback((command: ThinkingCommand, context: ThinkingContext = {}) => {
    if (command.type === "clearAll") {
      clear();
      return;
    }
    setDisplay((current) => {
      if (command.type === "clearSquare") {
        return { ...current, squares: withoutKey(current.squares, command.square) };
      }
      if (command.type === "clearArrow") {
        return { ...current, arrows: withoutKey(current.arrows, arrowKey(command.from, command.to)) };
      }

      const sequence = current.sequence + 1;
      const squares = prune(current.squares, sequence);
      const arrows = prune(current.arrows, sequence);
      if (command.type === "highlightSquare") {
        const key = command.square;
        squares[key] = { ...command, key: `${key}-${sequence}`, sequence };
      } else {
        const key = arrowKey(command.from, command.to);
        arrows[key] = {
          ...command,
          key: `${key}-${sequence}`,
          sequence,
          side: command.side ?? arrowSide(context.thinkingColor ?? null, context.sourceColor ?? null),
        };
      }
      return { sequence, squares, arrows };
    });
  }, [clear]);

  return { ...display, apply, clear };
}

export function ThinkingOverlay({
  enabled,
  orientation,
  sequence,
  squares,
  arrows,
}: {
  enabled: boolean;
  orientation: Color;
  sequence: number;
  squares: Record<string, SquareMark>;
  arrows: Record<string, ArrowMark>;
}) {
  const markerPrefix = useId().replaceAll(":", "");
  if (!enabled) return null;
  const visibleArrows = Object.values(arrows);
  return (
    <svg
      className="thinking-overlay"
      viewBox="0 0 100 100"
      preserveAspectRatio="none"
      aria-hidden="true"
    >
      <defs>
        {visibleArrows.map((mark) => {
          const color = arrowColor(mark.side);
          const opacity = annotationOpacity(mark.intensity, sequence - mark.sequence);
          return (
            <marker
              id={`${markerPrefix}-${mark.key}`}
              key={`${mark.key}-head`}
              markerWidth="4"
              markerHeight="4"
              refX="3"
              refY="2"
              orient="auto"
              markerUnits="strokeWidth"
            >
              <path d="M 0 0 L 4 2 L 0 4 z" fill={color} fillOpacity={opacity} />
            </marker>
          );
        })}
      </defs>
      {Object.values(squares).map((mark) => {
        const point = squareCenter(mark.square, orientation);
        return (
          <rect
            className="thinking-square"
            key={mark.key}
            x={point.x - 6.25}
            y={point.y - 6.25}
            width="12.5"
            height="12.5"
            fillOpacity={annotationOpacity(mark.intensity, sequence - mark.sequence)}
          />
        );
      })}
      {visibleArrows.map((mark) => {
        const color = arrowColor(mark.side);
        const opacity = annotationOpacity(mark.intensity, sequence - mark.sequence);
        const points = thinkingArrowPoints(mark.from, mark.to, orientation)
          .map((point) => `${point.x},${point.y}`)
          .join(" ");
        return (
          <polyline
            className={`thinking-arrow ${mark.side}`}
            key={mark.key}
            points={points}
            stroke={color}
            strokeOpacity={opacity}
            markerEnd={`url(#${markerPrefix}-${mark.key})`}
          />
        );
      })}
    </svg>
  );
}

function arrowKey(from: Square, to: Square): string {
  return `${from}-${to}`;
}

function arrowColor(side: ArrowSide): string {
  return side === "opponent" ? OPPONENT_ARROW_COLOR : OWN_ARROW_COLOR;
}

function withoutKey<T>(record: Record<string, T>, key: string): Record<string, T> {
  if (!(key in record)) return record;
  const next = { ...record };
  delete next[key];
  return next;
}

function prune<T extends Mark>(record: Record<string, T>, sequence: number): Record<string, T> {
  return Object.fromEntries(
    Object.entries(record).filter(([, mark]) => sequence - mark.sequence < MAX_NEWER_ANNOTATIONS),
  );
}
