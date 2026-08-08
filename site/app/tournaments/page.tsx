import type { Metadata } from "next";
import Link from "next/link";
import { getChatGPTUser } from "../chatgpt-auth";
import { listTournaments } from "../../lib/tournaments";
import { CreateTournamentForm } from "./create-tournament-form";
import { HistoryNav } from "../history/history-components";
import { formatDateTime } from "./tournament-nav";

export const dynamic = "force-dynamic";

export const metadata: Metadata = {
  title: "Tournaments · ChessGPT Arena",
  description: "Round-robin tournaments between submitted chess model packages.",
};

const STATUS_LABEL = {
  registration: "Registration open",
  running: "Running",
  completed: "Completed",
} as const;

export default async function TournamentsPage() {
  const [tournaments, user] = await Promise.all([
    listTournaments(),
    getChatGPTUser(),
  ]);

  return (
    <main className="history-page">
      <HistoryNav active="tournaments" />
      <header className="history-hero">
        <p className="eyebrow">ChessGPT arena</p>
        <h1>Tournaments</h1>
        <p>
          Every pair of entered models plays a fixed number of games from the standard
          starting position, on one pinned machine, under a per-move time limit. Ties
          share the title.
        </p>
      </header>

      {user ? <CreateTournamentForm /> : null}

      <section className="history-directory" aria-labelledby="tournament-list-title">
        <h2 className="sr-only" id="tournament-list-title">All tournaments</h2>
        <div className="history-list">
          {tournaments.length === 0 ? (
            <p className="history-empty">
              No tournaments yet.
              {user ? " Create one above." : " Sign in to create one."}
            </p>
          ) : tournaments.map((tournament) => (
            <Link
              className="history-directory-row tournament-row"
              key={tournament.id}
              href={`/tournaments/${tournament.id}`}
            >
              <div>
                <strong>{tournament.name}</strong>
                <code>
                  {tournament.gamesPerPair} games per pair ·{" "}
                  {tournament.moveTimeLimitMs} ms per move ·{" "}
                  {tournament.maxPlies} ply cap
                </code>
              </div>
              <span className={`tournament-status is-${tournament.status}`}>
                {STATUS_LABEL[tournament.status]}
              </span>
              <time dateTime={new Date(tournament.createdAt).toISOString()}>
                {formatDateTime(tournament.createdAt)}
              </time>
              <span aria-hidden="true">→</span>
            </Link>
          ))}
        </div>
      </section>
    </main>
  );
}
