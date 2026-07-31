import { getChatGPTUser } from "../../../../../chatgpt-auth";
import { withdrawEntry } from "../../../../../../lib/tournaments";
import { failure } from "../../../tournament-response";

export async function DELETE(
  _request: Request,
  context: { params: Promise<{ id: string; entryId: string }> },
) {
  try {
    const { id, entryId } = await context.params;
    await withdrawEntry(id, entryId, await getChatGPTUser());
    return new Response(null, { status: 204 });
  } catch (error) {
    return failure(error);
  }
}
