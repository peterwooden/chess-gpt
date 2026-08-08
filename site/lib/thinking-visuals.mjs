export const MAX_NEWER_ANNOTATIONS = 64;
const MAX_OPACITY = 0.8;

export function annotationOpacity(intensity, newerAnnotations) {
  const age = Math.min(MAX_NEWER_ANNOTATIONS, Math.max(0, newerAnnotations));
  const decay = (1 + Math.cos(Math.PI * age / MAX_NEWER_ANNOTATIONS)) / 2;
  return Number((Math.min(1, Math.max(0, intensity)) * MAX_OPACITY * decay).toFixed(6));
}

export function arrowSide(thinkingColor, sourceColor) {
  return sourceColor && thinkingColor && sourceColor !== thinkingColor ? "opponent" : "own";
}

export function thinkingArrowPoints(fromSquare, toSquare, orientation) {
  const from = squareCenter(fromSquare, orientation);
  const to = squareCenter(toSquare, orientation);
  const fileDistance = Math.abs(fromSquare.charCodeAt(0) - toSquare.charCodeAt(0));
  const rankDistance = Math.abs(Number(fromSquare[1]) - Number(toSquare[1]));
  if (!((fileDistance === 1 && rankDistance === 2) || (fileDistance === 2 && rankDistance === 1))) {
    return [from, to];
  }
  const bend = fileDistance > rankDistance
    ? { x: to.x, y: from.y }
    : { x: from.x, y: to.y };
  return [from, bend, to];
}

export function thinkingArrowShape(fromSquare, toSquare, orientation) {
  const points = thinkingArrowPoints(fromSquare, toSquare, orientation);
  const tip = points.at(-1);
  const previous = points.at(-2);
  const dx = tip.x - previous.x;
  const dy = tip.y - previous.y;
  const length = Math.hypot(dx, dy);
  const unit = { x: dx / length, y: dy / length };
  const base = roundPoint({ x: tip.x - unit.x * 4, y: tip.y - unit.y * 4 });
  const normal = { x: -unit.y * 3.2, y: unit.x * 3.2 };
  return {
    shaft: [...points.slice(0, -1), base],
    head: [
      tip,
      roundPoint({ x: base.x + normal.x, y: base.y + normal.y }),
      roundPoint({ x: base.x - normal.x, y: base.y - normal.y }),
    ],
  };
}

export function squareCenter(square, orientation) {
  const file = square.charCodeAt(0) - 97;
  const rank = Number(square[1]) - 1;
  return orientation === "w"
    ? { x: (file + 0.5) * 12.5, y: (7.5 - rank) * 12.5 }
    : { x: (7.5 - file) * 12.5, y: (rank + 0.5) * 12.5 };
}

function roundPoint(point) {
  return { x: Number(point.x.toFixed(6)), y: Number(point.y.toFixed(6)) };
}
