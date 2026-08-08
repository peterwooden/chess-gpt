import { HistoryError } from "../../../../lib/history";
import {
  getLiveGameResponse,
  publishLiveGame,
  type PublishLiveGameInput,
} from "../../../../lib/live-games";

export async function GET(
  request: Request,
  context: { params: Promise<{ id: string }> },
) {
  const { id } = await context.params;
  const rawAfter = new URL(request.url).searchParams.get("after");
  const parsedAfter = rawAfter === null ? undefined : Number(rawAfter);
  const after = parsedAfter !== undefined && Number.isFinite(parsedAfter)
    ? Math.max(0, Math.floor(parsedAfter))
    : undefined;
  const response = await getLiveGameResponse(id, after);
  if (!response.live && !response.completed) {
    return Response.json({ error: "Live game not found." }, { status: 404 });
  }
  return Response.json(response, { headers: { "Cache-Control": "no-store" } });
}

export async function PUT(
  request: Request,
  context: { params: Promise<{ id: string }> },
) {
  try {
    const { id } = await context.params;
    const contentLength = Number(request.headers.get("content-length"));
    if (Number.isFinite(contentLength) && contentLength > 80_000) {
      return Response.json({ error: "The live-game snapshot is too large." }, { status: 413 });
    }
    const input = await request.json() as PublishLiveGameInput;
    return Response.json({ live: await publishLiveGame(id, input) });
  } catch (error) {
    if (error instanceof SyntaxError) return Response.json({ error: "Invalid JSON." }, { status: 400 });
    const status = error instanceof HistoryError ? error.status : 500;
    return Response.json({
      error: error instanceof Error ? error.message : "The live game could not be updated.",
    }, { status });
  }
}
