import { recordAttempt } from "../../../../../lib/tournaments";
import { failure } from "../../tournament-response";

/**
 * Record an attempt at a scheduled game before it is played. Persisting this
 * server-side is what stops a page reload from resetting the counter, and so
 * what stops a package that reliably crashes the runner from stalling the
 * tournament forever.
 */
export async function POST(
  request: Request,
  context: { params: Promise<{ id: string }> },
) {
  try {
    const { id } = await context.params;
    const body = await request.json() as {
      pairKey?: unknown;
      gameIndex?: unknown;
      lastError?: unknown;
    };
    if (typeof body.pairKey !== "string" || !Number.isInteger(body.gameIndex)) {
      return Response.json({ error: "A pairing key and game index are required." }, { status: 400 });
    }
    const attempts = await recordAttempt(
      id,
      body.pairKey,
      body.gameIndex as number,
      typeof body.lastError === "string" ? body.lastError : null,
    );
    return Response.json({ attempts });
  } catch (error) {
    if (error instanceof SyntaxError) {
      return Response.json({ error: "Invalid JSON." }, { status: 400 });
    }
    return failure(error);
  }
}
