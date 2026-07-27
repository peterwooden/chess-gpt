import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { getPlayerProfile, listPlayerGames } from "../../../lib/history";
import { formatDate, HistoryNav } from "../../history/history-components";

export const dynamic = "force-dynamic";

export async function generateMetadata({ params }: { params: Promise<{ id: string }> }): Promise<Metadata> {
  const player = await getPlayerProfile((await params).id);
  return { title: player ? `${player.displayName} · ChessGPT history` : "Player not found · ChessGPT" };
}

export default async function PlayerPage({
  params,
  searchParams,
}: {
  params: Promise<{ id: string }>;
  searchParams: Promise<{ cursor?: string }>;
}) {
  const { id } = await params;
  const player = await getPlayerProfile(id);
  if (!player) notFound();
  const query = await searchParams;
  const history = await listPlayerGames(id, query.cursor);

  return (
    <main className="history-page profile-page">
      <HistoryNav />
      <header className="profile-hero">
        <Link className="profile-back" href={`/history?kind=${player.kind}`}>← All {player.kind === "model" ? "models" : "players"}</Link>
        <div className="profile-title">
          <div>
            <p className="eyebrow">{player.kind === "model" ? "Immutable model checkpoint" : "Signed-in player"}</p>
            <h1>{player.displayName}</h1>
          </div>
          <div className="profile-record" aria-label={`${player.wins} wins, ${player.draws} draws, ${player.losses} losses`}>
            <span><b>{player.wins}</b> wins</span>
            <span><b>{player.draws}</b> draws</span>
            <span><b>{player.losses}</b> losses</span>
          </div>
        </div>
        {player.kind === "model" ? (
          <dl className="model-identity-card">
            <div><dt>Repository</dt><dd><a href={`https://huggingface.co/${player.repository}`} rel="noreferrer">{player.repository}</a></dd></div>
            <div><dt>Commit</dt><dd><code>{player.commitSha}</code></dd></div>
            <div><dt>Manifest</dt><dd><code>{player.manifestSha256}</code></dd></div>
          </dl>
        ) : (
          <p className="human-identity-code">ChessGPT player <strong>{player.playerCode}</strong> · name frozen at first recorded game</p>
        )}
      </header>

      <section className="profile-games" aria-labelledby="games-title">
        <div className="profile-games-heading">
          <div><p className="eyebrow">Casual arena record</p><h2 id="games-title">Games</h2></div>
          <span>{player.games} total</span>
        </div>
        <div className="game-history-list">
          {history.games.length ? history.games.map((game) => {
            const isWhite = game.whitePlayerId === player.id;
            const selfPlay = game.whitePlayerId === player.id && game.blackPlayerId === player.id;
            const opponentId = isWhite ? game.blackPlayerId : game.whitePlayerId;
            const opponentName = selfPlay ? "Self-play" : isWhite ? game.blackName : game.whiteName;
            const outcome = selfPlay
              ? game.result === "1/2-1/2" ? "Draw" : game.result === "1-0" ? "White won" : "Black won"
              : outcomeFor(game.result, isWhite);
            return (
              <article className="game-history-row" key={game.id}>
                <div className={`result-mark ${outcome.toLocaleLowerCase().replace(" ", "-")}`}>{outcome}</div>
                <div className="game-opponent">
                  <span>{selfPlay ? "Against" : `${isWhite ? "White" : "Black"} against`}</span>
                  {opponentId && !selfPlay ? <Link href={`/players/${opponentId}`}>{opponentName}</Link> : <strong>{opponentName}</strong>}
                </div>
                <div className="game-meta">
                  <time dateTime={new Date(game.playedAt).toISOString()}>{formatDate(game.playedAt)}</time>
                  <span>{game.moveCount} {game.moveCount === 1 ? "move" : "moves"}</span>
                </div>
                <Link className="replay-link" href={`/arena?game=${game.id}`}>Replay →</Link>
              </article>
            );
          }) : <div className="history-empty"><span>00</span><p>No games recorded.</p></div>}
        </div>
        <nav className="history-pagination" aria-label="Game history pages">
          {query.cursor ? <Link href={`/players/${player.id}`}>First page</Link> : <span />}
          {history.nextCursor ? <Link href={`/players/${player.id}?cursor=${encodeURIComponent(history.nextCursor)}`}>Next page →</Link> : null}
        </nav>
      </section>
    </main>
  );
}

function outcomeFor(result: string, isWhite: boolean): "Win" | "Draw" | "Loss" {
  if (result === "1/2-1/2") return "Draw";
  return (result === "1-0") === isWhite ? "Win" : "Loss";
}
