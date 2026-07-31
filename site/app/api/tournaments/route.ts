import { getChatGPTUser } from "../../chatgpt-auth";
import { HistoryError } from "../../../lib/history";
import { createTournament, listTournaments, type TournamentConfig } from "../../../lib/tournaments";
import { failure } from "./tournament-response";

export async function GET() {
  try {
    return Response.json({ tournaments: await listTournaments() });
  } catch (error) {
    return failure(error);
  }
}

export async function POST(request: Request) {
  try {
    const payload = await request.json() as TournamentConfig;
    const tournament = await createTournament(payload, await getChatGPTUser());
    return Response.json({ tournament }, { status: 201 });
  } catch (error) {
    if (error instanceof SyntaxError) {
      return Response.json({ error: "Invalid JSON." }, { status: 400 });
    }
    return failure(error instanceof HistoryError ? error : error);
  }
}
