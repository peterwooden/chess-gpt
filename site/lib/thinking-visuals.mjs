const MAX_NEWER_ANNOTATIONS = 20;
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

export function squareCenter(square, orientation) {
  const file = square.charCodeAt(0) - 97;
  const rank = Number(square[1]) - 1;
  return orientation === "w"
    ? { x: (file + 0.5) * 12.5, y: (7.5 - rank) * 12.5 }
    : { x: (7.5 - file) * 12.5, y: (rank + 0.5) * 12.5 };
}
