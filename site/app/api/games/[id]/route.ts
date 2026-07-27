import { getPublicGame } from "../../../../lib/history";

export async function GET(
  _request: Request,
  context: { params: Promise<{ id: string }> },
) {
  const { id } = await context.params;
  const game = await getPublicGame(id);
  if (!game) return Response.json({ error: "Game not found." }, { status: 404 });
  return Response.json({ game });
}
