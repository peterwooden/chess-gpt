import { getChatGPTUser } from "../../../../chatgpt-auth";
import { listEntries, registerEntry, type RegisterEntryInput } from "../../../../../lib/tournaments";
import { failure } from "../../tournament-response";

export async function GET(
  _request: Request,
  context: { params: Promise<{ id: string }> },
) {
  try {
    const { id } = await context.params;
    return Response.json({ entries: await listEntries(id) });
  } catch (error) {
    return failure(error);
  }
}

export async function POST(
  request: Request,
  context: { params: Promise<{ id: string }> },
) {
  try {
    const { id } = await context.params;
    const payload = await request.json() as RegisterEntryInput;
    if (typeof payload.reference !== "string") {
      return Response.json({ error: "A model reference is required." }, { status: 400 });
    }
    const entry = await registerEntry(id, payload, await getChatGPTUser());
    return Response.json({ entry }, { status: 201 });
  } catch (error) {
    if (error instanceof SyntaxError) {
      return Response.json({ error: "Invalid JSON." }, { status: 400 });
    }
    return failure(error);
  }
}
