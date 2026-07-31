import { getChatGPTUser } from "../../../../chatgpt-auth";
import { claimRunner, heartbeat, type ClaimRunnerInput } from "../../../../../lib/tournaments";
import { failure } from "../../tournament-response";

/**
 * Claim the runner lease, or renew it. Renewal is a separate, cheap path so the
 * heartbeat does not repeat the pinning checks on every tick.
 */
export async function POST(
  request: Request,
  context: { params: Promise<{ id: string }> },
) {
  try {
    const { id } = await context.params;
    const body = await request.json() as ClaimRunnerInput & { renew?: boolean };
    if (typeof body.runnerId !== "string") {
      return Response.json({ error: "A runner identifier is required." }, { status: 400 });
    }
    if (body.renew) {
      return Response.json(await heartbeat(id, body.runnerId));
    }
    const tournament = await claimRunner(id, body, await getChatGPTUser());
    return Response.json({ tournament });
  } catch (error) {
    if (error instanceof SyntaxError) {
      return Response.json({ error: "Invalid JSON." }, { status: 400 });
    }
    return failure(error);
  }
}
