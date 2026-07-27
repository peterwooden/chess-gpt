import { getChatGPTUser } from "../../chatgpt-auth";
import { HistoryError, saveCompletedGame, type SaveGameInput } from "../../../lib/history";

export async function POST(request: Request) {
  try {
    const contentLength = Number(request.headers.get("content-length"));
    if (Number.isFinite(contentLength) && contentLength > 80_000) {
      return Response.json({ error: "The game submission is too large." }, { status: 413 });
    }
    const payload = await request.json() as SaveGameInput;
    const game = await saveCompletedGame(payload, await getChatGPTUser());
    return Response.json({ game }, { status: 201 });
  } catch (error) {
    if (error instanceof SyntaxError) return Response.json({ error: "Invalid JSON." }, { status: 400 });
    const status = error instanceof HistoryError ? error.status : 500;
    const message = error instanceof Error ? error.message : "The game could not be saved.";
    return Response.json({ error: message }, { status });
  }
}
