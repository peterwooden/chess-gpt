import type { Metadata } from "next";
import Link from "next/link";
import { listModels } from "../../lib/history";
import { modelRepositoryHref } from "../arena/hugging-face-reference.mjs";
import { formatDate, HistoryNav } from "../history/history-components";

export const dynamic = "force-dynamic";

export const metadata: Metadata = {
  title: "Models · ChessGPT Arena",
  description: "Discover compatible chess-model repositories and every immutable version observed by the ChessGPT arena.",
};

type ModelSearchParams = { q?: string; sort?: string; cursor?: string };

export default async function ModelsPage({ searchParams }: { searchParams: Promise<ModelSearchParams> }) {
  const params = await searchParams;
  const sort = params.sort === "name" || params.sort === "games" || params.sort === "versions"
    ? params.sort
    : "recent";
  const search = params.q?.trim() ?? "";
  const directory = await listModels({ sort, search, cursor: params.cursor });
  const next = new URLSearchParams({ sort });
  if (search) next.set("q", search);
  if (directory.nextCursor) next.set("cursor", directory.nextCursor);

  return (
    <main className="history-page models-page">
      <HistoryNav active="models" />
      <header className="history-hero">
        <p className="eyebrow">ChessGPT model catalog</p>
        <h1>Models</h1>
        <p>Discover compatible Hugging Face repositories and the exact immutable versions first seen by this arena.</p>
      </header>

      <section className="history-directory" aria-labelledby="models-title">
        <form className="history-filters model-filters" action="/models" method="get">
          <label>
            <span>Search</span>
            <input type="search" name="q" defaultValue={search} placeholder="Name, repository, or commit SHA" />
          </label>
          <label>
            <span>Sort</span>
            <select name="sort" defaultValue={sort}>
              <option value="recent">Recent activity</option>
              <option value="name">Name</option>
              <option value="games">Games played</option>
              <option value="versions">Versions observed</option>
            </select>
          </label>
          <button type="submit">Apply</button>
        </form>

        <div className="history-list-heading model-list-heading" id="models-title">
          <span>Model repository</span><span>Record</span><span>Versions</span><span>Activity</span>
        </div>
        <div className="history-list model-directory-list">
          {directory.models.length ? directory.models.map((model) => (
            <Link className="history-directory-row model-directory-row" href={modelRepositoryHref(model.repository)} key={model.repository}>
              <div>
                <strong>{model.displayName}</strong>
                <code>{model.repository}</code>
                <small className="latest-model-reference" title={`${model.repository}@${model.latestCommitSha}`}>
                  Latest · {model.repository}@{model.latestCommitSha}
                </small>
              </div>
              <div className="wdl" aria-label={`${model.wins} wins, ${model.draws} draws, ${model.losses} losses`}>
                <span><b>{model.wins}</b> W</span><span><b>{model.draws}</b> D</span><span><b>{model.losses}</b> L</span>
                <small>{model.games} {model.games === 1 ? "game" : "games"}</small>
              </div>
              <div className="version-count"><strong>{model.versions}</strong><span>{model.versions === 1 ? "version" : "versions"}</span></div>
              <time dateTime={new Date(model.lastPlayedAt).toISOString()}>{formatDate(model.lastPlayedAt)}</time>
              <span className="row-arrow" aria-hidden="true">→</span>
            </Link>
          )) : (
            <div className="history-empty"><span>00</span><p>{search ? "No matching models were found." : "No compatible models have been observed yet."}</p></div>
          )}
        </div>
        <nav className="history-pagination" aria-label="Model catalog pages">
          {params.cursor ? <Link href={`/models?${new URLSearchParams({ sort, ...(search ? { q: search } : {}) })}`}>First page</Link> : <span />}
          {directory.nextCursor ? <Link href={`/models?${next}`}>Next page →</Link> : null}
        </nav>
      </section>
    </main>
  );
}
