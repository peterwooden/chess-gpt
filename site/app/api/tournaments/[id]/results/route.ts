import { getTournamentResults } from "../../../../../lib/tournaments";
import { failure } from "../../tournament-response";

export async function GET(
  _request: Request,
  context: { params: Promise<{ id: string }> },
) {
  try {
    const { id } = await context.params;
    const results = await getTournamentResults(id);
    // The runner polls this between games, so the full game list is omitted.
    return Response.json({
      table: results.table,
      shared: results.shared,
      distinctGamesByPair: results.distinctGamesByPair,
      playedCount: results.games.length,
    });
  } catch (error) {
    return failure(error);
  }
}
