import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { getChatGPTUser } from "../../chatgpt-auth";
import { ensureHumanPlayer } from "../../../lib/history";
import { getTournament, getTournamentResults, isAdministrator } from "../../../lib/tournaments";
import { HistoryNav } from "../../history/history-components";
import { formatDateTime, formatScore } from "../tournament-nav";
import { TournamentAdminControls } from "./admin-controls";
import { RegisterEntryPanel } from "./register-entry-panel";

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
  const [results, administrator] = await Promise.all([
    getTournamentResults(id),
    isAdministrator(user),
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
          {tournament.maxPlies} ply cap · {entries.length} entries · {games.length} of{" "}
          {scheduled} games played
        </p>
      </header>

      {administrator ? (
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

      {games.length > 0 ? (
        <section className="tournament-panel" aria-labelledby="standings-title">
          <h2 id="standings-title">Standings</h2>
          {shared ? (
            <p className="tournament-note">
              The leaders are level on points and share the title. No further games are
              played to separate them.
            </p>
          ) : null}
          <table className="tournament-table">
            <thead>
              <tr>
                <th scope="col">#</th>
                <th scope="col">Model</th>
                <th scope="col">Points</th>
                <th scope="col">W</th>
                <th scope="col">D</th>
                <th scope="col">L</th>
                <th scope="col">Games</th>
                <th scope="col">Score</th>
              </tr>
            </thead>
            <tbody>
              {table.map((row) => (
                <tr key={row.entryId}>
                  <td>{row.rank}</td>
                  <td>{row.displayName}</td>
                  <td><strong>{formatScore(row.points)}</strong></td>
                  <td>{row.wins}</td>
                  <td>{row.draws}</td>
                  <td>{row.losses}</td>
                  <td>{row.games}</td>
                  <td>{row.games > 0 ? `${Math.round((row.points / row.games) * 100)}%` : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>

          {Object.keys(distinctGamesByPair).length > 0 ? (
            <details className="tournament-details">
              <summary>Distinct games per pairing</summary>
              <p className="tournament-note">
                Standings count every game. A pairing that produced few distinct games
                replayed the same moves, which is what happens when both packages are
                deterministic.
              </p>
              <ul>
                {Object.entries(distinctGamesByPair).map(([key, count]) => {
                  const played = games.filter((game) => game.pairKey === key).length;
                  const names = key.split(":")
                    .map((entryId) => entries.find((entry) => entry.id === entryId)?.displayName ?? "unknown");
                  return (
                    <li key={key}>
                      {names.join(" v ")}: <strong>{count}</strong> distinct of {played} played
                    </li>
                  );
                })}
              </ul>
            </details>
          ) : null}
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
                <span>{game.result}</span>
                <time dateTime={new Date(game.recordedAt).toISOString()}>
                  {formatDateTime(game.recordedAt)}
                </time>
                <span aria-hidden="true">→</span>
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
