import type { Color, Square } from "chess.js";

export type ArrowSide = "own" | "opponent";

export const MAX_NEWER_ANNOTATIONS: 64;
export type ThinkingPoint = { x: number; y: number };

export function annotationOpacity(intensity: number, newerAnnotations: number): number;
export function arrowSide(thinkingColor: Color | null, sourceColor: Color | null): ArrowSide;
export function thinkingArrowPoints(
  fromSquare: Square,
  toSquare: Square,
  orientation: Color,
): ThinkingPoint[];
export function thinkingArrowShape(
  fromSquare: Square,
  toSquare: Square,
  orientation: Color,
): { shaft: ThinkingPoint[]; head: ThinkingPoint[] };
export function squareCenter(square: Square, orientation: Color): ThinkingPoint;
