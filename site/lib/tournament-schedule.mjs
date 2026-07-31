/**
 * The tournament schedule.
 *
 * Every scheduled game is derived from the entry list and games_per_pair alone,
 * so "what plays next" is a pure function of the tournament row. That is what
 * lets the runner resume by subtracting the games already recorded rather than
 * inferring its position from history.
 */

/**
 * Build the pairing key for two entries. Entry ids are sorted so the key does
 * not depend on which side happens to be playing white in a given game.
 *
 * @param {string} entryA
 * @param {string} entryB
 * @returns {string}
 */
export function pairKey(entryA, entryB) {
  return entryA < entryB ? `${entryA}:${entryB}` : `${entryB}:${entryA}`;
}

/**
 * Every unordered pairing of entries, in a deterministic order.
 *
 * @param {ReadonlyArray<{ id: string }>} entries
 * @returns {Array<{ key: string, first: string, second: string }>}
 */
export function pairings(entries) {
  const ids = entries.map((entry) => entry.id).sort();
  const result = [];
  for (let i = 0; i < ids.length; i += 1) {
    for (let j = i + 1; j < ids.length; j += 1) {
      result.push({ key: pairKey(ids[i], ids[j]), first: ids[i], second: ids[j] });
    }
  }
  return result;
}

/**
 * The complete schedule.
 *
 * Games are interleaved rather than pair-major — round one plays one game of
 * every pairing, then round two — because interruption is a first-class case.
 * An interleaved run stopped at any point leaves every pairing with a
 * comparable number of games and balanced colors; a pair-major run stopped
 * early yields complete data on some pairings and nothing at all on others.
 *
 * Colors alternate with the game index, so odd-length partial runs are off
 * balance by at most one game per pairing.
 *
 * @param {ReadonlyArray<{ id: string }>} entries
 * @param {number} gamesPerPair
 * @param {{ order?: "interleaved" | "pair-major" }} [options]
 * @returns {Array<{ pairKey: string, gameIndex: number, whiteEntryId: string, blackEntryId: string }>}
 */
export function buildSchedule(entries, gamesPerPair, options = {}) {
  if (!Number.isInteger(gamesPerPair) || gamesPerPair < 1) {
    throw new Error("gamesPerPair must be a positive integer.");
  }
  const pairs = pairings(entries);
  const scheduled = [];

  const emit = (pair, gameIndex) => {
    const whiteFirst = gameIndex % 2 === 0;
    scheduled.push({
      pairKey: pair.key,
      gameIndex,
      whiteEntryId: whiteFirst ? pair.first : pair.second,
      blackEntryId: whiteFirst ? pair.second : pair.first,
    });
  };

  if (options.order === "pair-major") {
    for (const pair of pairs) {
      for (let gameIndex = 0; gameIndex < gamesPerPair; gameIndex += 1) emit(pair, gameIndex);
    }
    return scheduled;
  }

  for (let gameIndex = 0; gameIndex < gamesPerPair; gameIndex += 1) {
    for (const pair of pairs) emit(pair, gameIndex);
  }
  return scheduled;
}

/**
 * The scheduled games that still need playing, in order.
 *
 * @param {ReturnType<typeof buildSchedule>} schedule
 * @param {Iterable<{ pairKey: string, gameIndex: number }>} recorded
 * @returns {ReturnType<typeof buildSchedule>}
 */
export function remainingGames(schedule, recorded) {
  const done = new Set();
  for (const game of recorded) done.add(`${game.pairKey}#${game.gameIndex}`);
  return schedule.filter((game) => !done.has(`${game.pairKey}#${game.gameIndex}`));
}

/**
 * Standings over recorded tournament games.
 *
 * Points only: a balanced round robin needs no rating fit, and ties share the
 * title rather than being separated by further games. Distinct game counts are
 * reported so a pairing that produced two distinct games out of two hundred is
 * visibly different from one that produced two hundred.
 *
 * @param {ReadonlyArray<{ id: string, displayName: string }>} entries
 * @param {ReadonlyArray<{
 *   whiteEntryId: string, blackEntryId: string, result: string,
 *   pairKey: string, moveList?: string,
 * }>} games
 */
export function standings(entries, games) {
  const rows = new Map(entries.map((entry) => [entry.id, {
    entryId: entry.id,
    displayName: entry.displayName,
    games: 0,
    wins: 0,
    draws: 0,
    losses: 0,
    points: 0,
  }]));
  const distinct = new Map();

  for (const game of games) {
    const white = rows.get(game.whiteEntryId);
    const black = rows.get(game.blackEntryId);
    if (!white || !black) continue;
    white.games += 1;
    black.games += 1;
    if (game.result === "1-0") {
      white.wins += 1; white.points += 1; black.losses += 1;
    } else if (game.result === "0-1") {
      black.wins += 1; black.points += 1; white.losses += 1;
    } else {
      white.draws += 1; white.points += 0.5;
      black.draws += 1; black.points += 0.5;
    }

    if (game.moveList !== undefined) {
      const seen = distinct.get(game.pairKey) ?? new Set();
      seen.add(game.moveList);
      distinct.set(game.pairKey, seen);
    }
  }

  const table = [...rows.values()].sort((a, b) => (
    b.points - a.points
    || b.wins - a.wins
    || a.displayName.localeCompare(b.displayName)
  ));

  // Equal points share a rank. Ties share the title, so nothing here breaks them.
  let rank = 0;
  let previousPoints = Number.NaN;
  const ranked = table.map((row, index) => {
    if (row.points !== previousPoints) {
      rank = index + 1;
      previousPoints = row.points;
    }
    return { ...row, rank };
  });

  return {
    table: ranked,
    shared: ranked.filter((row) => row.rank === 1).length > 1,
    distinctGamesByPair: Object.fromEntries(
      [...distinct.entries()].map(([key, set]) => [key, set.size]),
    ),
  };
}
