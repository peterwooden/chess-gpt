import { getChatGPTUser } from "../../../../chatgpt-auth";
import { setTournamentStatus, type TournamentStatus } from "../../../../../lib/tournaments";
import { failure } from "../../tournament-response";

export async function POST(
  request: Request,
  context: { params: Promise<{ id: string }> },
) {
  try {
    const { id } = await context.params;
    const body = await request.json() as { status?: TournamentStatus };
    if (body.status !== "registration" && body.status !== "running" && body.status !== "completed") {
      return Response.json({ error: "Unknown tournament status." }, { status: 400 });
    }
    const tournament = await setTournamentStatus(id, body.status, await getChatGPTUser());
    return Response.json({ tournament });
  } catch (error) {
    return failure(error);
  }
}
