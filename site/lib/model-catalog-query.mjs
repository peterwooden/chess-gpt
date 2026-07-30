export function buildModelDirectoryQuery({ search = "", sort = "recent", cursor = null, limit = 25 }) {
  const normalizedSort = ["recent", "name", "games", "versions"].includes(sort) ? sort : "recent";
  const where = ["1 = 1"];
  const bindings = [];
  if (search) {
    where.push(`(displayName LIKE ? COLLATE NOCASE OR repository LIKE ? COLLATE NOCASE
      OR EXISTS (SELECT 1 FROM model_versions searched WHERE searched.repository = modelDirectory.repository
        AND searched.commit_sha LIKE ? COLLATE NOCASE))`);
    const pattern = `%${search}%`;
    bindings.push(pattern, pattern, pattern);
  }
  let order = "lastPlayedAt DESC, repository DESC";
  if (cursor && normalizedSort === "recent") {
    where.push("(lastPlayedAt < ? OR (lastPlayedAt = ? AND repository < ?))");
    bindings.push(Number(cursor.value), Number(cursor.value), cursor.id);
  } else if (cursor && normalizedSort === "name") {
    where.push("(lower(displayName) > ? OR (lower(displayName) = ? AND repository > ?))");
    bindings.push(cursor.value, cursor.value, cursor.id);
  } else if (cursor && (normalizedSort === "games" || normalizedSort === "versions")) {
    where.push(`(${normalizedSort} < ? OR (${normalizedSort} = ? AND repository < ?))`);
    bindings.push(Number(cursor.value), Number(cursor.value), cursor.id);
  }
  if (normalizedSort === "name") order = "lower(displayName) ASC, repository ASC";
  else if (normalizedSort === "games") order = "games DESC, repository DESC";
  else if (normalizedSort === "versions") order = "versions DESC, repository DESC";
  bindings.push(limit);
  return {
    sql: `${modelDirectoryCte()}
      SELECT * FROM modelDirectory
      WHERE ${where.join(" AND ")}
      ORDER BY ${order}
      LIMIT ?`,
    bindings,
  };
}

export function buildModelProfileQuery() {
  return `${modelDirectoryCte()} SELECT * FROM modelDirectory WHERE repository = ?`;
}

export const MODEL_VERSIONS_SQL = `SELECT
    mv.player_id AS playerId, p.display_name AS displayName, mv.repository,
    mv.commit_sha AS commitSha, mv.manifest_sha256 AS manifestSha256,
    mv.first_seen_at AS firstSeenAt, p.last_played_at AS lastPlayedAt,
    COUNT(g.id) AS games,
    COALESCE(SUM(CASE WHEN (g.seat = 'w' AND g.result = '1-0')
      OR (g.seat = 'b' AND g.result = '0-1') THEN 1 ELSE 0 END), 0) AS wins,
    COALESCE(SUM(CASE WHEN g.result = '1/2-1/2' THEN 1 ELSE 0 END), 0) AS draws,
    COALESCE(SUM(CASE WHEN (g.seat = 'w' AND g.result = '0-1')
      OR (g.seat = 'b' AND g.result = '1-0') THEN 1 ELSE 0 END), 0) AS losses
  FROM model_versions mv
  JOIN players p ON p.id = mv.player_id
  LEFT JOIN (
    SELECT id, result, white_player_id AS player_id, 'w' AS seat FROM games WHERE white_player_id IS NOT NULL
    UNION ALL
    SELECT id, result, black_player_id AS player_id, 'b' AS seat FROM games WHERE black_player_id IS NOT NULL
  ) g ON g.player_id = mv.player_id
  WHERE mv.repository = ?
  GROUP BY mv.player_id
  ORDER BY mv.first_seen_at DESC, mv.player_id DESC`;

function modelDirectoryCte() {
  return `WITH modelGames AS (
      SELECT id, result, white_player_id AS player_id, 'w' AS seat FROM games WHERE white_player_id IS NOT NULL
      UNION ALL
      SELECT id, result, black_player_id AS player_id, 'b' AS seat FROM games WHERE black_player_id IS NOT NULL
    ), modelDirectory AS (
      SELECT
        m.repository, m.display_name AS displayName, m.first_seen_at AS firstSeenAt,
        MAX(p.last_played_at) AS lastPlayedAt,
        (SELECT latest.commit_sha FROM model_versions latest
          WHERE latest.repository = m.repository
          ORDER BY latest.first_seen_at DESC, latest.player_id DESC LIMIT 1) AS latestCommitSha,
        (SELECT latest.first_seen_at FROM model_versions latest
          WHERE latest.repository = m.repository
          ORDER BY latest.first_seen_at DESC, latest.player_id DESC LIMIT 1) AS latestFirstSeenAt,
        COUNT(DISTINCT mv.player_id) AS versions,
        COUNT(g.id) AS games,
        COALESCE(SUM(CASE WHEN (g.seat = 'w' AND g.result = '1-0')
          OR (g.seat = 'b' AND g.result = '0-1') THEN 1 ELSE 0 END), 0) AS wins,
        COALESCE(SUM(CASE WHEN g.result = '1/2-1/2' THEN 1 ELSE 0 END), 0) AS draws,
        COALESCE(SUM(CASE WHEN (g.seat = 'w' AND g.result = '0-1')
          OR (g.seat = 'b' AND g.result = '1-0') THEN 1 ELSE 0 END), 0) AS losses
      FROM models m
      JOIN model_versions mv ON mv.repository = m.repository
      JOIN players p ON p.id = mv.player_id
      LEFT JOIN modelGames g ON g.player_id = mv.player_id
      GROUP BY m.repository
    )`;
}
