import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { getModelProfile, listModelVersions, listPlayerGames } from "../../../../lib/history";
import { modelPageHref } from "../../../arena/hugging-face-reference.mjs";
import { formatDate, HistoryNav } from "../../../history/history-components";
import { ReferenceActions } from "../../reference-actions";
import { selectModelVersion } from "../../model-page.mjs";

export const dynamic = "force-dynamic";

type ModelParams = { owner: string; repository: string };

export async function generateMetadata({ params }: { params: Promise<ModelParams> }): Promise<Metadata> {
  const { owner, repository: name } = await params;
  const model = await getModelProfile(`${owner}/${name}`);
  return { title: model ? `${model.displayName} · ChessGPT models` : "Model not found · ChessGPT" };
}

export default async function ModelPage({
  params,
  searchParams,
}: {
  params: Promise<ModelParams>;
  searchParams: Promise<{ version?: string; cursor?: string }>;
}) {
  const { owner, repository: name } = await params;
  const repository = `${owner}/${name}`;
  const [model, versions] = await Promise.all([getModelProfile(repository), listModelVersions(repository)]);
  if (!model || !versions.length) notFound();
  const query = await searchParams;
  const selected = selectModelVersion(versions, query.version);
  if (!selected) notFound();
  const reference = `${repository}@${selected.commitSha}`;
  const history = await listPlayerGames(selected.playerId, query.cursor);

  return (
    <main className="history-page model-profile-page">
      <HistoryNav active="models" />
      <header className="profile-hero model-profile-hero">
        <Link className="profile-back" href="/models">← All models</Link>
        <div className="profile-title">
          <div><p className="eyebrow">Model repository</p><h1>{model.displayName}</h1><code className="model-repository">{repository}</code></div>
          <div className="profile-record" aria-label={`${model.wins} wins, ${model.draws} draws, ${model.losses} losses`}>
            <span><b>{model.wins}</b> wins</span><span><b>{model.draws}</b> draws</span><span><b>{model.losses}</b> losses</span>
          </div>
        </div>
        <div className="selected-reference">
          <div><span>Selected immutable reference</span><code>{reference}</code></div>
          <ReferenceActions reference={reference} />
        </div>
      </header>

      <section className="model-version-section" aria-labelledby="versions-title">
        <div className="profile-games-heading">
          <div><p className="eyebrow">First seen by ChessGPT</p><h2 id="versions-title">Version history</h2></div>
          <span>{versions.length} observed</span>
        </div>
        <ol className="version-timeline">
          {versions.map((version) => {
            const versionReference = `${repository}@${version.commitSha}`;
            const active = version.commitSha === selected.commitSha;
            return (
              <li className={active ? "selected" : ""} id={`version-${version.commitSha}`} key={version.commitSha}>
                <div className="version-date"><time dateTime={new Date(version.firstSeenAt).toISOString()}>{formatDate(version.firstSeenAt)}</time><i /></div>
                <article>
                  <header><div><strong>{version.displayName}</strong><code>{versionReference}</code></div>{active ? <span>Selected</span> : null}</header>
                  <dl>
                    <div><dt>Manifest SHA-256</dt><dd><code>{version.manifestSha256}</code></dd></div>
                    <div><dt>Record</dt><dd>{version.wins} W · {version.draws} D · {version.losses} L</dd></div>
                    <div><dt>Games</dt><dd>{version.games}</dd></div>
                  </dl>
                  <div className="version-actions">
                    {!active ? <Link href={modelPageHref(versionReference)}>View version</Link> : <span />}
                    <ReferenceActions reference={versionReference} compact />
                  </div>
                </article>
              </li>
            );
          })}
        </ol>
      </section>

      <section className="profile-games model-version-games" aria-labelledby="games-title">
        <div className="profile-games-heading">
          <div><p className="eyebrow">Selected version</p><h2 id="games-title">Games</h2></div>
          <span>{selected.games} total</span>
        </div>
        <div className="game-history-list">
          {history.games.length ? history.games.map((game) => {
            const isWhite = game.whitePlayerId === selected.playerId;
            const selfPlay = game.whitePlayerId === selected.playerId && game.blackPlayerId === selected.playerId;
            const opponentId = isWhite ? game.blackPlayerId : game.whitePlayerId;
            const opponentName = selfPlay ? "Self-play" : isWhite ? game.blackName : game.whiteName;
            const outcome = selfPlay
              ? game.result === "1/2-1/2" ? "Draw" : game.result === "1-0" ? "White won" : "Black won"
              : outcomeFor(game.result, isWhite);
            return (
              <article className="game-history-row" key={game.id}>
                <div className={`result-mark ${outcome.toLocaleLowerCase().replace(" ", "-")}`}>{outcome}</div>
                <div className="game-opponent"><span>{selfPlay ? "Against" : `${isWhite ? "White" : "Black"} against`}</span>{opponentId && !selfPlay ? <Link href={`/players/${opponentId}`}>{opponentName}</Link> : <strong>{opponentName}</strong>}</div>
                <div className="game-meta"><time dateTime={new Date(game.playedAt).toISOString()}>{formatDate(game.playedAt)}</time><span>{game.moveCount} {game.moveCount === 1 ? "move" : "moves"}</span></div>
                <Link className="replay-link" href={`/arena?game=${game.id}`}>Replay →</Link>
              </article>
            );
          }) : <div className="history-empty"><span>00</span><p>No games recorded for this version.</p></div>}
        </div>
        <nav className="history-pagination" aria-label="Version game pages">
          {query.cursor ? <Link href={modelPageHref(reference)}>First page</Link> : <span />}
          {history.nextCursor ? <Link href={`${modelPageHref(reference)}&cursor=${encodeURIComponent(history.nextCursor)}`}>Next page →</Link> : null}
        </nav>
      </section>
    </main>
  );
}

function outcomeFor(result: string, isWhite: boolean): "Win" | "Draw" | "Loss" {
  if (result === "1/2-1/2") return "Draw";
  return (result === "1-0") === isWhite ? "Win" : "Loss";
}
