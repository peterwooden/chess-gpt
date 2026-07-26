const MODEL_PARAMETERS = { a: "modelA", b: "modelB" };

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
