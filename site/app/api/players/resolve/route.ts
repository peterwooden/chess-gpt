import { ensureModelPlayer, HistoryError } from "../../../../lib/history";

export async function POST(request: Request) {
  try {
    const body = await request.json() as { reference?: unknown };
    if (typeof body.reference !== "string") {
      return Response.json({ error: "A model reference is required." }, { status: 400 });
    }
    const profile = await ensureModelPlayer(body.reference);
    return Response.json(profile);
  } catch (error) {
    const status = error instanceof HistoryError ? error.status : 500;
    const message = error instanceof HistoryError ? error.message : "The model history profile could not be resolved.";
    return Response.json({ error: message }, { status });
  }
}
