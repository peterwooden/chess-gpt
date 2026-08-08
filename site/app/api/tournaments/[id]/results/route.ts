import { getTournamentResults } from "../../../../../lib/tournaments";
import { getTournamentLiveGame } from "../../../../../lib/live-games";
import { failure } from "../../tournament-response";

export async function GET(
  _request: Request,
  context: { params: Promise<{ id: string }> },
) {
  try {
    const { id } = await context.params;
    const [results, liveGame] = await Promise.all([
      getTournamentResults(id),
      getTournamentLiveGame(id),
    ]);
    const scheduledCount = (
      results.entries.length * (results.entries.length - 1) / 2
    ) * results.tournament.gamesPerPair;
    // The runner polls this between games, so the full game list is omitted.
    return Response.json({
      table: results.table,
      shared: results.shared,
      distinctGamesByPair: results.distinctGamesByPair,
      playedCount: results.games.length,
      scheduledCount,
      status: results.tournament.status,
      liveGame,
    });
  } catch (error) {
    return failure(error);
  }
}
