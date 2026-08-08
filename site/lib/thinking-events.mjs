const SQUARE_PATTERN = /^[a-h][1-8]$/;
const COMMANDS_PER_WINDOW = 128;
const WINDOW_MS = 500;

export function normalizeThinkingCommand(value) {
  if (!value || typeof value !== "object" || typeof value.type !== "string") return null;
  if (value.type === "highlightSquare") {
    if (!isSquare(value.square)) return null;
    const drawing = normalizeDrawing(value);
    return drawing && { type: value.type, square: value.square, ...drawing };
  }
  if (value.type === "drawArrow") {
    if (!isSquare(value.from) || !isSquare(value.to) || value.from === value.to) return null;
    const drawing = normalizeDrawing(value);
    const side = value.side === undefined ? {} : normalizeArrowSide(value.side);
    return drawing && side && { type: value.type, from: value.from, to: value.to, ...drawing, ...side };
  }
  if (value.type === "clearSquare") {
    return isSquare(value.square) ? { type: value.type, square: value.square } : null;
  }
  if (value.type === "clearArrow") {
    return isSquare(value.from) && isSquare(value.to) && value.from !== value.to
      ? { type: value.type, from: value.from, to: value.to }
      : null;
  }
  return value.type === "clearAll" ? { type: value.type } : null;
}

export function createThinkingCommandLimiter(now) {
  let windowStartedAt = now();
  let accepted = 0;
  return {
    accept() {
      const current = now();
      if (current - windowStartedAt >= WINDOW_MS) {
        windowStartedAt = current;
        accepted = 0;
      }
      if (accepted >= COMMANDS_PER_WINDOW) return false;
      accepted += 1;
      return true;
    },
  };
}

function normalizeDrawing(value) {
  const intensity = value.intensity === undefined ? 1 : value.intensity;
  const fadeMs = value.fadeMs === undefined ? 500 : value.fadeMs;
  if (!Number.isFinite(intensity) || intensity < 0 || intensity > 1) return null;
  if (!Number.isFinite(fadeMs) || fadeMs < 0) return null;
  return { intensity, fadeMs };
}

function normalizeArrowSide(value) {
  return value === "own" || value === "opponent" ? { side: value } : null;
}

function isSquare(value) {
  return typeof value === "string" && SQUARE_PATTERN.test(value);
}
