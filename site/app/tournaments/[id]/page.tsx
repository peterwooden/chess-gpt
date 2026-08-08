import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { getChatGPTUser } from "../../chatgpt-auth";
import { ensureHumanPlayer } from "../../../lib/history";
import { getTournamentLiveGame } from "../../../lib/live-games";
import { getTournament, getTournamentResults, isAdministrator, isTournamentManager } from "../../../lib/tournaments";
import { HistoryNav } from "../../history/history-components";
import { formatDateTime } from "../tournament-nav";
import { TournamentAdminControls } from "./admin-controls";
import { RegisterEntryPanel } from "./register-entry-panel";
import { TournamentBroadcast } from "./tournament-broadcast";

export const dynamic = "force-dynamic";

export async function generateMetadata(
  { params }: { params: Promise<{ id: string }> },
): Promise<Metadata> {
  const { id } = await params;
  const tournament = await getTournament(id);
  return { title: `${tournament?.name ?? "Tournament"} · ChessGPT Arena` };
}

const STATUS_LABEL = {
  registration: "Registration open",
  running: "Running",
  completed: "Completed",
} as const;

export default async function TournamentPage(
  { params }: { params: Promise<{ id: string }> },
) {
  const { id } = await params;
  const tournament = await getTournament(id);
  if (!tournament) notFound();

  const user = await getChatGPTUser();
  const [results, liveGame, administrator, manager] = await Promise.all([
    getTournamentResults(id),
    getTournamentLiveGame(id),
    isAdministrator(user),
    isTournamentManager(tournament, user),
  ]);
  const viewer = user ? await ensureHumanPlayer(user) : null;
  const { entries, games, table, shared, distinctGamesByPair, runnerChanges } = results;
  const scheduled = (entries.length * (entries.length - 1) / 2) * tournament.gamesPerPair;

  return (
    <main className="history-page">
      <HistoryNav active="tournaments" />
      <header className="history-hero">
        <p className="eyebrow">{STATUS_LABEL[tournament.status]}</p>
        <h1>{tournament.name}</h1>
        <p>
          {tournament.gamesPerPair} games per pair · {tournament.moveTimeLimitMs} ms per move ·{" "}
          {tournament.openingBook ? "sampled openings" : "standard start"} ·{" "}
          {entries.length} entries · {games.length} of{" "}
          {scheduled} games played
        </p>
      </header>

      {manager ? (
        <TournamentAdminControls
          tournamentId={tournament.id}
          status={tournament.status}
        />
      ) : null}

      {tournament.status === "running" ? (
        <section className="tournament-panel">
          <p className="tournament-runner-callout">
            This tournament is ready to play.{" "}
            <Link href={`/tournaments/${tournament.id}/run`}>Open the runner</Link> on the
            machine that will host it, and leave that machine idle until it finishes.
          </p>
        </section>
      ) : null}

      {runnerChanges.length > 0 ? (
        <section className="tournament-panel">
          <p className="tournament-warning">
            <strong>Played across more than one machine.</strong> Under a per-move time
            limit this is not a clean result, and it is recorded here permanently.
          </p>
          <ul className="tournament-runner-changes">
            {runnerChanges.map((change, index) => (
              <li key={index}>
                {formatDateTime(change.at)} — moved from{" "}
                “{change.fromLabel ?? "unknown machine"}” to “{change.toLabel}”, authorised by{" "}
                {change.authorisedBy}
              </li>
            ))}
          </ul>
        </section>
      ) : null}

      <TournamentBroadcast
        tournamentId={tournament.id}
        initial={{
          table,
          shared,
          playedCount: games.length,
          scheduledCount: scheduled,
          status: tournament.status,
          liveGame,
        }}
      />

      {Object.keys(distinctGamesByPair).length > 0 ? (
        <section className="tournament-panel">
          <details className="tournament-details">
            <summary>Distinct games per pairing</summary>
            <p className="tournament-note">
              Standings count every game. A pairing that produced few distinct games
              replayed the same moves, which is what happens when both packages are deterministic.
            </p>
            <ul>
              {Object.entries(distinctGamesByPair).map(([key, count]) => {
                const played = games.filter((game) => game.pairKey === key).length;
                const names = key.split(":")
                  .map((entryId) => entries.find((entry) => entry.id === entryId)?.displayName ?? "unknown");
                return <li key={key}>{names.join(" v ")}: <strong>{Number(count)}</strong> distinct of {played} played</li>;
              })}
            </ul>
          </details>
        </section>
      ) : null}

      <RegisterEntryPanel
        tournamentId={tournament.id}
        status={tournament.status}
        entries={entries.map((entry) => ({
          id: entry.id,
          displayName: entry.displayName,
          reference: entry.reference,
          ownerPlayerId: entry.ownerPlayerId,
          verifiedAt: entry.verifiedAt,
          smokeMedianMs: entry.smokeMedianMs,
          packageBytes: entry.packageBytes,
        }))}
        viewerPlayerId={viewer?.id ?? null}
        administrator={administrator}
        signedIn={Boolean(user)}
      />

      {games.length > 0 ? (
        <section className="tournament-panel" aria-labelledby="games-title">
          <h2 id="games-title">Games</h2>
          <div className="history-list">
            {games.slice(0, 200).map((game) => (
              <Link className="history-directory-row tournament-game-row" key={game.id} href={`/arena?game=${game.id}`}>
                <div>
                  <strong>{game.whiteName} v {game.blackName}</strong>
                  <code>{game.termination} · {game.moveCount} moves</code>
                </div>
                <span className="tournament-game-result">{game.result}</span>
                <time dateTime={new Date(game.recordedAt).toISOString()}>
                  {formatDateTime(game.recordedAt)}
                </time>
                <span className="row-arrow" aria-hidden="true">→</span>
              </Link>
            ))}
          </div>
          {games.length > 200 ? (
            <p className="tournament-note">Showing the first 200 of {games.length} games.</p>
          ) : null}
        </section>
      ) : null}
    </main>
  );
}
