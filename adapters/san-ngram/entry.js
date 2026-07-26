function asNgramState(bytes) {
  let state;
  try {
    state = JSON.parse(new TextDecoder().decode(bytes));
  } catch {
    throw new Error("The model artifact is not valid JSON.");
  }
  if (
    !state
    || state.format_version !== 1
    || state.model_type !== "san_backoff_ngram"
    || !Number.isInteger(state.order)
    || !state.ngrams
    || !state.side_counts
  ) {
    throw new Error("The model artifact is not a supported SAN backoff n-gram state.");
  }
  return state;
}

function bestLegal(pairs, legalMoves) {
  let chosen = null;
  let chosenCount = -1;
  for (const pair of pairs) {
    if (!Array.isArray(pair) || pair.length !== 2) continue;
    const [san, count] = pair;
    if (!legalMoves.has(san)) continue;
    if (count > chosenCount || (count === chosenCount && san > (chosen ?? ""))) {
      chosen = san;
      chosenCount = count;
    }
  }
  return chosen;
}

export async function loadPackage({ artifacts }) {
  const modelBytes = artifacts.get("model");
  if (!(modelBytes instanceof Uint8Array)) {
    throw new Error("The package requires a Uint8Array artifact named model.");
  }
  const state = asNgramState(modelBytes);

  return {
    async newGame() {
      return {
        async chooseMove({ history, legalMoves }) {
          if (!Array.isArray(history) || !Array.isArray(legalMoves) || legalMoves.length === 0) {
            throw new Error("chooseMove requires SAN history and at least one legal SAN move.");
          }
          const legal = new Set(legalMoves);
          for (let order = Math.min(state.order, history.length); order > 0; order -= 1) {
            const context = history.slice(-order).join("\t");
            const chosen = bestLegal(state.ngrams[String(order)]?.[context] ?? [], legal);
            if (chosen !== null) return chosen;
          }
          const side = String(history.length % 2);
          const chosen = bestLegal(state.side_counts[side] ?? [], legal);
          if (chosen !== null) return chosen;
          return [...legal].sort()[0];
        },
        async dispose() {},
      };
    },
    async dispose() {},
  };
}
