import { normalizeThinkingCommand } from "./thinking-events.mjs";

const MAX_EVENTS_PER_BATCH = 256;
const MAX_THINKING_COMMANDS_PER_BATCH = 128;

export function createLiveGameEventSequencer(now) {
  const startedAt = now();
  let nextSeq = 1;
  let nextBatchIndex = 1;
  let lastOffsetMs = 0;
  let pendingThinkingCommands = 0;
  let pending = [];
  return {
    currentOffsetMs() {
      return Math.max(lastOffsetMs, Math.round(now() - startedAt));
    },
    record(payload, capturedOffsetMs) {
      if (payload?.type === "thinking.command" && pendingThinkingCommands >= MAX_THINKING_COMMANDS_PER_BATCH) {
        return null;
      }
      const candidate = capturedOffsetMs === undefined
        ? Math.round(now() - startedAt)
        : Math.round(capturedOffsetMs);
      const offsetMs = Math.max(lastOffsetMs, Number.isFinite(candidate) ? candidate : lastOffsetMs);
      const event = { seq: nextSeq, offsetMs, payload };
      nextSeq += 1;
      lastOffsetMs = offsetMs;
      pending.push(event);
      if (payload?.type === "thinking.command") pendingThinkingCommands += 1;
      return event;
    },
    flush() {
      if (pending.length === 0) return null;
      const events = pending;
      pending = [];
      pendingThinkingCommands = 0;
      const batch = {
        batchIndex: nextBatchIndex,
        firstSeq: events[0].seq,
        lastSeq: events.at(-1).seq,
        events,
      };
      nextBatchIndex += 1;
      return batch;
    },
  };
}

export function normalizeLiveGameEventBatch(value) {
  if (!value || typeof value !== "object") return null;
  if (!isPositiveInteger(value.batchIndex) || !isPositiveInteger(value.firstSeq)
      || !isPositiveInteger(value.lastSeq) || !Array.isArray(value.events)
      || value.events.length === 0 || value.events.length > MAX_EVENTS_PER_BATCH) return null;
  if (value.lastSeq - value.firstSeq + 1 !== value.events.length) return null;
  let lastOffsetMs = -1;
  let thinkingCount = 0;
  const events = [];
  for (let index = 0; index < value.events.length; index += 1) {
    const candidate = value.events[index];
    if (!candidate || typeof candidate !== "object"
        || candidate.seq !== value.firstSeq + index
        || !Number.isFinite(candidate.offsetMs) || candidate.offsetMs < lastOffsetMs) return null;
    const payload = normalizePayload(candidate.payload);
    if (!payload) return null;
    if (payload.type === "thinking.command") thinkingCount += 1;
    if (thinkingCount > MAX_THINKING_COMMANDS_PER_BATCH) return null;
    lastOffsetMs = candidate.offsetMs;
    events.push({ seq: candidate.seq, offsetMs: Math.round(candidate.offsetMs), payload });
  }
  return {
    batchIndex: value.batchIndex,
    firstSeq: value.firstSeq,
    lastSeq: value.lastSeq,
    events,
  };
}

function normalizePayload(value) {
  if (!value || typeof value !== "object" || typeof value.type !== "string") return null;
  if (value.type === "game.started") return { type: value.type };
  if (value.type === "turn.started") {
    if (typeof value.turnId !== "string" || value.turnId.length === 0 || value.turnId.length > 80
        || !isPositiveInteger(value.ply) || (value.color !== "w" && value.color !== "b")) return null;
    return { type: value.type, turnId: value.turnId, ply: value.ply, color: value.color };
  }
  if (value.type === "thinking.command") {
    if (typeof value.turnId !== "string" || value.turnId.length === 0 || value.turnId.length > 80) return null;
    const command = normalizeThinkingCommand(value.command);
    return command ? { type: value.type, turnId: value.turnId, command } : null;
  }
  if (value.type === "move.played") {
    if (value.turnId !== null && (typeof value.turnId !== "string" || value.turnId.length > 80)) return null;
    if (!isPositiveInteger(value.ply) || typeof value.san !== "string" || value.san.length > 24
        || (value.color !== "w" && value.color !== "b") || !isSquare(value.from) || !isSquare(value.to)
        || typeof value.actor !== "string" || value.actor.length === 0 || value.actor.length > 120
        || !Number.isFinite(value.elapsedMs) || value.elapsedMs < 0) return null;
    return {
      type: value.type,
      turnId: value.turnId,
      ply: value.ply,
      san: value.san,
      color: value.color,
      from: value.from,
      to: value.to,
      actor: value.actor,
      elapsedMs: Math.round(value.elapsedMs),
    };
  }
  if (value.type === "game.updated") {
    return value.update && typeof value.update === "object"
      ? { type: value.type, update: value.update }
      : null;
  }
  return null;
}

function isPositiveInteger(value) {
  return Number.isInteger(value) && value > 0;
}

function isSquare(value) {
  return typeof value === "string" && /^[a-h][1-8]$/.test(value);
}
