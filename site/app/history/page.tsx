import type { Metadata } from "next";
import Link from "next/link";
import { listPlayers } from "../../lib/history";
import { formatDate, HistoryNav } from "./history-components";

export const dynamic = "force-dynamic";

export const metadata: Metadata = {
  title: "Game history · ChessGPT Arena",
  description: "Browse human players, immutable chess-model checkpoints, and their recorded arena games.",
};

type HistorySearchParams = {
  kind?: string;
  q?: string;
  sort?: string;
  cursor?: string;
};

export default async function HistoryPage({
  searchParams,
}: {
  searchParams: Promise<HistorySearchParams>;
}) {
  const params = await searchParams;
  const kind = params.kind === "human" ? "human" : "model";
  const sort = params.sort === "name" || params.sort === "games" ? params.sort : "recent";
  const search = params.q?.trim() ?? "";
  const directory = await listPlayers({ kind, sort, search, cursor: params.cursor });
  const next = new URLSearchParams({ kind, sort });
  if (search) next.set("q", search);
  if (directory.nextCursor) next.set("cursor", directory.nextCursor);

  return (
    <main className="history-page">
      <HistoryNav />
      <header className="history-hero">
        <p className="eyebrow">ChessGPT arena ledger</p>
        <h1>Game history</h1>
        <p>Browse signed-in human players and the exact immutable model checkpoints they faced.</p>
      </header>

      <section className="history-directory" aria-labelledby="directory-title">
        <nav className="history-tabs" aria-label="History type">
          <Link className={kind === "model" ? "active" : ""} href="/history?kind=model">Models</Link>
          <Link className={kind === "human" ? "active" : ""} href="/history?kind=human">Players</Link>
        </nav>
        <form className="history-filters" action="/history" method="get">
          <input type="hidden" name="kind" value={kind} />
          <label>
            <span>Search</span>
            <input
              type="search"
              name="q"
              defaultValue={search}
              placeholder={kind === "model" ? "Name, repository, or SHA" : "Name or player code"}
            />
          </label>
          <label>
            <span>Sort</span>
            <select name="sort" defaultValue={sort}>
              <option value="recent">Recently active</option>
              <option value="name">Name</option>
              <option value="games">Games played</option>
            </select>
          </label>
          <button type="submit">Apply</button>
        </form>

        <div className="history-list-heading" id="directory-title">
          <span>{kind === "model" ? "Model checkpoint" : "Player"}</span>
          <span>Record</span>
          <span>Last game</span>
        </div>
        <div className="history-list">
          {directory.players.length ? directory.players.map((player) => (
            <Link className="history-directory-row" href={`/players/${player.id}`} key={player.id}>
              <div>
                <strong>{player.displayName}</strong>
                <code>
                  {player.kind === "model"
                    ? `${player.repository}@${player.commitSha?.slice(0, 10)}…`
                    : `Player ${player.playerCode}`}
                </code>
              </div>
              <div className="wdl" aria-label={`${player.wins} wins, ${player.draws} draws, ${player.losses} losses`}>
                <span><b>{player.wins}</b> W</span>
                <span><b>{player.draws}</b> D</span>
                <span><b>{player.losses}</b> L</span>
                <small>{player.games} {player.games === 1 ? "game" : "games"}</small>
              </div>
              <time dateTime={new Date(player.lastPlayedAt).toISOString()}>{formatDate(player.lastPlayedAt)}</time>
              <span className="row-arrow" aria-hidden="true">→</span>
            </Link>
          )) : (
            <div className="history-empty">
              <span>00</span>
              <p>{search ? "No matching histories were found." : `No ${kind === "model" ? "model" : "player"} games have been recorded yet.`}</p>
            </div>
          )}
        </div>
        <nav className="history-pagination" aria-label="Directory pages">
          {params.cursor ? <Link href={`/history?${new URLSearchParams({ kind, sort, ...(search ? { q: search } : {}) })}`}>First page</Link> : <span />}
          {directory.nextCursor ? <Link href={`/history?${next}`}>Next page →</Link> : null}
        </nav>
      </section>
    </main>
  );
}
