import { getRunnerPlan } from "../../../../../lib/tournaments";
import { failure } from "../../tournament-response";

export async function GET(
  _request: Request,
  context: { params: Promise<{ id: string }> },
) {
  try {
    const { id } = await context.params;
    return Response.json(await getRunnerPlan(id));
  } catch (error) {
    return failure(error);
  }
}
