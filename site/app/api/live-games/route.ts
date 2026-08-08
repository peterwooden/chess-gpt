import { HistoryError } from "../../../lib/history";
import { openLiveGame, type OpenLiveGameInput } from "../../../lib/live-games";

export async function POST(request: Request) {
  try {
    const contentLength = Number(request.headers.get("content-length"));
    if (Number.isFinite(contentLength) && contentLength > 8_000) {
      return Response.json({ error: "The live-game description is too large." }, { status: 413 });
    }
    const input = await request.json() as OpenLiveGameInput;
    return Response.json({ live: await openLiveGame(input) }, { status: 201 });
  } catch (error) {
    if (error instanceof SyntaxError) return Response.json({ error: "Invalid JSON." }, { status: 400 });
    const status = error instanceof HistoryError ? error.status : 500;
    return Response.json({
      error: error instanceof Error ? error.message : "The live game could not be opened.",
    }, { status });
  }
}
