import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { requireChatGPTUser } from "../../../chatgpt-auth";
import { getTournament } from "../../../../lib/tournaments";
import { TournamentNav } from "../../tournament-nav";
import { TournamentRunner } from "./tournament-runner";

export const dynamic = "force-dynamic";

export const metadata: Metadata = {
  title: "Run tournament · ChessGPT Arena",
};

export default async function RunTournamentPage(
  { params }: { params: Promise<{ id: string }> },
) {
  const { id } = await params;
  // The runner requires a signed-in user but not an administrator, so the link
  // can be handed to whoever's machine is hosting the tournament.
  await requireChatGPTUser(`/tournaments/${id}/run`);
  const tournament = await getTournament(id);
  if (!tournament) notFound();

  return (
    <main className="history-page">
      <TournamentNav />
      <header className="history-hero">
        <p className="eyebrow">Tournament runner</p>
        <h1>{tournament.name}</h1>
        <p>
          Games run one at a time on this machine. Leave it idle and awake until the
          tournament finishes — anything else competing for the processor eats into
          each package&rsquo;s {tournament.moveTimeLimitMs} ms move budget.
        </p>
      </header>
      <TournamentRunner tournamentId={tournament.id} />
    </main>
  );
}
