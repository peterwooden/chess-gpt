import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { getLiveGameResponse } from "../../../lib/live-games";
import { LiveGameViewer } from "./live-game-viewer";

export const dynamic = "force-dynamic";

export async function generateMetadata(
  { params }: { params: Promise<{ id: string }> },
): Promise<Metadata> {
  const { id } = await params;
  const { live, completed } = await getLiveGameResponse(id);
  const game = live ?? completed;
  return {
    title: game
      ? `${game.whiteName} v ${game.blackName} · ChessGPT Live`
      : "Live game · ChessGPT Arena",
    description: "Follow a ChessGPT arena game move by move.",
  };
}

export default async function WatchGamePage(
  { params }: { params: Promise<{ id: string }> },
) {
  const { id } = await params;
  const initial = await getLiveGameResponse(id);
  if (!initial.live && !initial.completed) notFound();
  return <LiveGameViewer gameId={id} initial={initial} />;
}

