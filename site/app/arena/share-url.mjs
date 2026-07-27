const MODEL_PARAMETERS = { a: "modelA", b: "modelB" };
const PGN_PARAMETER = "pgn";

/**
 * Return the model defaults explicitly carried by a shared arena URL.
 * A missing slot stays empty so device-local history cannot alter the shared setup.
 *
 * @param {string | URL} input
 * @returns {{ a: string, b: string } | null}
 */
export function readSharedModelReferences(input) {
  const url = new URL(input);
  if (!url.searchParams.has(MODEL_PARAMETERS.a) && !url.searchParams.has(MODEL_PARAMETERS.b)) {
    return null;
  }
  return {
    a: (url.searchParams.get(MODEL_PARAMETERS.a) ?? "").trim(),
    b: (url.searchParams.get(MODEL_PARAMETERS.b) ?? "").trim(),
  };
}

/**
 * Return the same arena URL with one successfully loaded model recorded for sharing.
 *
 * @param {string | URL} input
 * @param {"a" | "b"} slot
 * @param {string} reference
 * @returns {string}
 */
export function withSharedModelReference(input, slot, reference) {
  const url = new URL(input);
  url.searchParams.set(MODEL_PARAMETERS[slot], reference.trim());
  return url.toString();
}

/**
 * Return the PGN carried by a shared arena URL, if present.
 *
 * @param {string | URL} input
 * @returns {string | null}
 */
export function readSharedPgn(input) {
  const pgn = new URL(input).searchParams.get(PGN_PARAMETER);
  if (!pgn?.trim()) return null;
  // chess.js 1.4 rejects digits in custom PGN tag names. Preserve links
  // generated before the arena renamed this tag.
  return pgn.replace(/^\[Player1Color\s/m, "[PlayerOneColor ");
}

/**
 * Build a stable share URL from the current arena location.
 * Empty model slots are removed and PGN is included only for a full-game share.
 *
 * @param {string | URL} input
 * @param {{ a: string, b: string, pgn?: string | null }} state
 * @returns {string}
 */
export function buildArenaShareUrl(input, state) {
  const url = new URL(input);
  for (const slot of ["a", "b"]) {
    const reference = state[slot].trim();
    if (reference) url.searchParams.set(MODEL_PARAMETERS[slot], reference);
    else url.searchParams.delete(MODEL_PARAMETERS[slot]);
  }
  if (state.pgn?.trim()) url.searchParams.set(PGN_PARAMETER, state.pgn);
  else url.searchParams.delete(PGN_PARAMETER);
  return url.toString();
}
