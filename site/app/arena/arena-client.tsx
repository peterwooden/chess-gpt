"use client";

import { Chess, type Color, type PieceSymbol, type Square } from "chess.js";
import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  loadBrowserModel,
  type BrowserChessModel,
  type LoadProgress,
} from "./model";

const MODEL_URLS_KEY = "chess-gpt:arena-model-urls-v1";
const FILES = ["a", "b", "c", "d", "e", "f", "g", "h"] as const;
const RANKS = [8, 7, 6, 5, 4, 3, 2, 1] as const;
const PIECES: Record<Color, Record<PieceSymbol, string>> = {
  w: { p: "♙", n: "♘", b: "♗", r: "♖", q: "♕", k: "♔" },
  b: { p: "♟", n: "♞", b: "♝", r: "♜", q: "♛", k: "♚" },
};

type ModelSlot = {
  reference: string;
  phase: "idle" | "loading" | "ready" | "error";
  progress: LoadProgress | null;
  model: BrowserChessModel | null;
  error: string | null;
};

type PlayMode = "human" | "models";

type MoveRecord = {
  ply: number;
  san: string;
  actor: string;
  source: string;
  elapsedMs: number | null;
  from: Square;
  to: Square;
};

type PromotionChoice = {
  from: Square;
  to: Square;
  pieces: PieceSymbol[];
};

function emptySlot(): ModelSlot {
  return { reference: "", phase: "idle", progress: null, model: null, error: null };
}

export default function ArenaClient() {
  const gameRef = useRef(new Chess());
  const gameEpoch = useRef(0);
  const loadEpoch = useRef({ a: 0, b: 0 });
  const loadedModels = useRef<{ a: BrowserChessModel | null; b: BrowserChessModel | null }>({
    a: null,
    b: null,
  });
  const [modelA, setModelA] = useState<ModelSlot>(emptySlot);
  const [modelB, setModelB] = useState<ModelSlot>(emptySlot);
  const [mode, setMode] = useState<PlayMode>("human");
  const [humanColor, setHumanColor] = useState<Color>("w");
  const [running, setRunning] = useState(false);
  const [thinking, setThinking] = useState<string | null>(null);
  const [delayMs, setDelayMs] = useState(650);
  const [gameStarted, setGameStarted] = useState(false);
  const [, setGameVersion] = useState(0);
  const [moves, setMoves] = useState<MoveRecord[]>([]);
  const [selectedSquare, setSelectedSquare] = useState<Square | null>(null);
  const [promotion, setPromotion] = useState<PromotionChoice | null>(null);
  const [gameError, setGameError] = useState<string | null>(null);

  useEffect(() => {
    const saved = window.localStorage.getItem(MODEL_URLS_KEY);
    if (!saved) return;
    try {
      const urls = JSON.parse(saved) as { a?: string; b?: string };
      setModelA((current) => ({ ...current, reference: urls.a ?? "" }));
      setModelB((current) => ({ ...current, reference: urls.b ?? "" }));
    } catch {
      window.localStorage.removeItem(MODEL_URLS_KEY);
    }
  }, []);

  useEffect(() => {
    window.localStorage.setItem(
      MODEL_URLS_KEY,
      JSON.stringify({ a: modelA.reference, b: modelB.reference }),
    );
  }, [modelA.reference, modelB.reference]);

  useEffect(() => {
    loadedModels.current = { a: modelA.model, b: modelB.model };
  }, [modelA.model, modelB.model]);

  useEffect(() => () => {
    loadEpoch.current.a += 1;
    loadEpoch.current.b += 1;
    void loadedModels.current.a?.dispose();
    void loadedModels.current.b?.dispose();
  }, []);

  const game = gameRef.current;
  const history = game.history();
  const targetSquares = selectedSquare
    ? new Set(game.moves({ square: selectedSquare, verbose: true }).map((move) => move.to))
    : new Set<Square>();
  const latestMove = moves.at(-1);
  const orientation: Color = mode === "human" ? humanColor : "w";
  const boardRanks = orientation === "w" ? RANKS : [...RANKS].reverse();
  const boardFiles = orientation === "w" ? FILES : [...FILES].reverse();
  const status = describeGame(game, running, thinking);

  const loadModel = useCallback(
    async (slot: "a" | "b") => {
      const current = slot === "a" ? modelA : modelB;
      const setSlot = slot === "a" ? setModelA : setModelB;
      if (!current.reference.trim()) {
        setSlot((value) => ({ ...value, phase: "error", error: "Enter a model reference first." }));
        return;
      }
      setRunning(false);
      gameEpoch.current += 1;
      const requestId = ++loadEpoch.current[slot];
      await current.model?.dispose();
      setSlot((value) => ({ ...value, phase: "loading", progress: null, model: null, error: null }));
      try {
        const loaded = await loadBrowserModel(current.reference, (progress) => {
          if (loadEpoch.current[slot] !== requestId) return;
          setSlot((value) => ({ ...value, progress }));
        });
        if (loadEpoch.current[slot] !== requestId) {
          await loaded.dispose();
          return;
        }
        setSlot((value) => ({
          ...value,
          phase: "ready",
          progress: null,
          model: loaded,
          error: null,
        }));
      } catch (error) {
        if (loadEpoch.current[slot] !== requestId) return;
        setSlot((value) => ({
          ...value,
          phase: "error",
          progress: null,
          model: null,
          error: error instanceof Error ? error.message : "The model could not be loaded.",
        }));
      }
    },
    [modelA, modelB],
  );

  function updateReference(slot: "a" | "b", reference: string) {
    const setSlot = slot === "a" ? setModelA : setModelB;
    setSlot((current) => ({ ...current, reference, error: null }));
  }

  function beginGame(nextMode: PlayMode) {
    if (!modelA.model) {
      setGameError("Load Model A before starting a game.");
      return;
    }
    if (nextMode === "models" && !modelB.model) {
      setGameError("Load Model B before starting model-versus-model play.");
      return;
    }
    gameEpoch.current += 1;
    gameRef.current = new Chess();
    setMode(nextMode);
    setMoves([]);
    setSelectedSquare(null);
    setPromotion(null);
    setThinking(null);
    setGameError(null);
    setGameStarted(true);
    setRunning(true);
    setGameVersion((value) => value + 1);
  }

  const playModelMove = useCallback(
    async (model: BrowserChessModel, actor: string) => {
      const epoch = gameEpoch.current;
      const activeGame = gameRef.current;
      if (activeGame.isGameOver()) return;
      setThinking(actor);
      setGameError(null);
      const started = performance.now();
      try {
        const prediction = await model.predict(activeGame.history(), activeGame.moves());
        if (gameEpoch.current !== epoch) return;
        const move = activeGame.move(prediction.san);
        const elapsedMs = performance.now() - started;
        setMoves((current) => [
          ...current,
          {
            ply: current.length + 1,
            san: move.san,
            actor,
            source: prediction.source,
            elapsedMs,
            from: move.from,
            to: move.to,
          },
        ]);
        setGameVersion((value) => value + 1);
        if (activeGame.isGameOver()) setRunning(false);
      } catch (error) {
        if (gameEpoch.current !== epoch) return;
        setRunning(false);
        setGameError(error instanceof Error ? error.message : `${actor} failed to return a legal move.`);
      } finally {
        if (gameEpoch.current === epoch) setThinking(null);
      }
    },
    [],
  );

  const activeModel = (() => {
    if (!running || game.isGameOver()) return null;
    if (mode === "models") {
      return game.turn() === "w"
        ? modelA.model && { model: modelA.model, actor: modelA.model.info.name }
        : modelB.model && { model: modelB.model, actor: modelB.model.info.name };
    }
    if (game.turn() === humanColor) return null;
    return modelA.model && { model: modelA.model, actor: modelA.model.info.name };
  })();

  useEffect(() => {
    if (!activeModel || thinking) return;
    const timer = window.setTimeout(
      () => void playModelMove(activeModel.model, activeModel.actor),
      mode === "models" ? delayMs : 180,
    );
    return () => window.clearTimeout(timer);
  }, [activeModel, delayMs, mode, playModelMove, thinking]);

  function chooseSquare(square: Square) {
    if (mode !== "human" || !running || thinking || game.turn() !== humanColor) return;
    const piece = game.get(square);
    if (!selectedSquare) {
      if (piece?.color === humanColor) setSelectedSquare(square);
      return;
    }
    if (piece?.color === humanColor) {
      setSelectedSquare(square);
      return;
    }
    const candidates = game
      .moves({ square: selectedSquare, verbose: true })
      .filter((move) => move.to === square);
    if (candidates.length === 0) {
      setSelectedSquare(null);
      return;
    }
    const promotionPieces = candidates
      .map((move) => move.promotion)
      .filter((pieceSymbol): pieceSymbol is PieceSymbol => Boolean(pieceSymbol));
    if (promotionPieces.length > 0) {
      setPromotion({ from: selectedSquare, to: square, pieces: promotionPieces });
      return;
    }
    playHumanMove(selectedSquare, square);
  }

  function playHumanMove(from: Square, to: Square, promotionPiece?: PieceSymbol) {
    try {
      const move = game.move({ from, to, promotion: promotionPiece });
      setMoves((current) => [
        ...current,
        {
          ply: current.length + 1,
          san: move.san,
          actor: "Human",
          source: "board input",
          elapsedMs: null,
          from: move.from,
          to: move.to,
        },
      ]);
      setSelectedSquare(null);
      setPromotion(null);
      setGameError(null);
      setGameVersion((value) => value + 1);
      if (game.isGameOver()) setRunning(false);
    } catch {
      setSelectedSquare(null);
      setPromotion(null);
      setGameError("That move is not legal in the current position.");
    }
  }

  async function stepOnce() {
    if (thinking || game.isGameOver()) return;
    const candidate =
      mode === "models"
        ? game.turn() === "w"
          ? modelA.model && { model: modelA.model, actor: modelA.model.info.name }
          : modelB.model && { model: modelB.model, actor: modelB.model.info.name }
        : game.turn() !== humanColor
          ? modelA.model && { model: modelA.model, actor: modelA.model.info.name }
          : null;
    if (candidate) await playModelMove(candidate.model, candidate.actor);
  }

  return (
    <main className="arena-page">
      <nav className="arena-nav" aria-label="Arena navigation">
        <Link href="/" className="wordmark">CGPT / LAB</Link>
        <span>Browser arena · local inference</span>
      </nav>

      <header className="arena-hero">
        <div>
          <p className="eyebrow">Playable model harness · zero server inference</p>
          <h1>Two URLs.<br />One board.</h1>
        </div>
        <p>
          Download revisioned Hugging Face artifacts directly into this browser. Play Model A yourself,
          or load Model B and watch a complete game unfold move by move.
        </p>
      </header>

      <section className="model-rack" aria-labelledby="model-rack-title">
        <div className="arena-section-heading">
          <div>
            <p className="eyebrow">01 · Select players</p>
            <h2 id="model-rack-title">Load the learned state.</h2>
          </div>
          <p>
            Prefer <code>owner/repository@commit</code>. A branch such as <code>main</code> works for
            exploration, but it is mutable and cannot identify a tournament submission.
          </p>
        </div>
        <div className="model-grid">
          <ModelLoader
            label="Model A"
            role="Human opponent · White in autoplay"
            slot={modelA}
            onReference={(reference) => updateReference("a", reference)}
            onLoad={() => void loadModel("a")}
          />
          <ModelLoader
            label="Model B"
            role="Optional · Black in autoplay"
            slot={modelB}
            onReference={(reference) => updateReference("b", reference)}
            onLoad={() => void loadModel("b")}
          />
        </div>
        <p className="contract-note">
          Accepted artifacts: a direct <code>model.json.gz</code> SAN n-gram checkpoint, or a repository
          containing <code>browser/manifest.json</code> under the safe <code>chess-gpt-browser-v1</code> contract.
        </p>
      </section>

      <section className="play-lab" aria-labelledby="board-title">
        <div className="play-toolbar">
          <div>
            <p className="eyebrow">02 · Run inference</p>
            <h2 id="board-title">Watch every decision.</h2>
          </div>
          <div className="play-options">
            <label>
              Human side
              <select
                value={humanColor}
                onChange={(event) => setHumanColor(event.target.value as Color)}
                disabled={gameStarted && !game.isGameOver()}
              >
                <option value="w">White</option>
                <option value="b">Black</option>
              </select>
            </label>
            <label>
              Autoplay pace
              <select value={delayMs} onChange={(event) => setDelayMs(Number(event.target.value))}>
                <option value={120}>Fast · 0.12 s</option>
                <option value={650}>Readable · 0.65 s</option>
                <option value={1500}>Study · 1.5 s</option>
              </select>
            </label>
          </div>
          <div className="play-actions">
            <button type="button" className="arena-primary" onClick={() => beginGame("human")}>
              Human vs A
            </button>
            <button type="button" onClick={() => beginGame("models")} disabled={!modelB.model}>
              A vs B
            </button>
            <button
              type="button"
              onClick={() => setRunning((value) => !value)}
              disabled={!gameStarted || game.isGameOver()}
            >
              {running ? "Pause" : "Resume"}
            </button>
            <button type="button" onClick={() => void stepOnce()} disabled={running || Boolean(thinking)}>
              Next move
            </button>
          </div>
        </div>

        <div className="arena-workbench">
          <div>
            <div className="game-status" aria-live="polite">
              <span>{mode === "human" ? "Human match" : "Model match"}</span>
              <strong>{status}</strong>
              <small>{history.length} plies · {Math.ceil(history.length / 2)} moves</small>
            </div>
            {gameError ? <div className="arena-error" role="alert">{gameError}</div> : null}
            <div className={`chessboard orientation-${orientation}`} role="grid" aria-label="Chess board">
              {boardRanks.flatMap((rank) =>
                boardFiles.map((file) => {
                  const square = `${file}${rank}` as Square;
                  const piece = game.get(square);
                  const light = (FILES.indexOf(file) + rank) % 2 === 1;
                  const selected = square === selectedSquare;
                  const target = targetSquares.has(square);
                  const last = square === latestMove?.from || square === latestMove?.to;
                  return (
                    <button
                      type="button"
                      role="gridcell"
                      aria-label={`${square}${piece ? ` ${piece.color === "w" ? "white" : "black"} ${pieceName(piece.type)}` : " empty"}`}
                      className={`board-square ${light ? "light" : "dark"}${selected ? " selected" : ""}${target ? " target" : ""}${last ? " last" : ""}`}
                      onClick={() => chooseSquare(square)}
                      key={square}
                    >
                      {piece ? <span className={`piece ${piece.color}`}>{PIECES[piece.color][piece.type]}</span> : null}
                      {(orientation === "w" ? rank === 1 : rank === 8) ? <small className="file-label">{file}</small> : null}
                      {(orientation === "w" ? file === "a" : file === "h") ? <small className="rank-label">{rank}</small> : null}
                    </button>
                  );
                }),
              )}
            </div>
            {promotion ? (
              <div className="promotion-picker" role="dialog" aria-label="Choose promotion piece">
                <strong>Promote pawn to</strong>
                <div>
                  {promotion.pieces.map((piece) => (
                    <button
                      type="button"
                      onClick={() => playHumanMove(promotion.from, promotion.to, piece)}
                      aria-label={`Promote to ${pieceName(piece)}`}
                      key={piece}
                    >
                      {PIECES[humanColor][piece]}
                    </button>
                  ))}
                  <button type="button" className="promotion-cancel" onClick={() => setPromotion(null)}>Cancel</button>
                </div>
              </div>
            ) : null}
          </div>

          <aside className="move-console" aria-label="Game progress">
            <header>
              <div><span>Live record</span><strong>{thinking ? `${thinking} is thinking` : running ? "Game running" : "Game paused"}</strong></div>
              <i className={thinking ? "pulse active" : "pulse"} aria-hidden="true" />
            </header>
            {moves.length === 0 ? (
              <div className="empty-record">
                <span>—</span>
                <p>Load a model and begin a game. SAN moves and inference timing will appear here.</p>
              </div>
            ) : (
              <ol className="move-record">
                {moves.map((move) => (
                  <li className={move.ply === moves.length ? "current" : ""} key={`${move.ply}-${move.san}`}>
                    <span>{move.ply}</span>
                    <strong>{move.san}</strong>
                    <div><b>{move.actor}</b><small>{move.source}{move.elapsedMs === null ? "" : ` · ${Math.round(move.elapsedMs)} ms`}</small></div>
                  </li>
                ))}
              </ol>
            )}
          </aside>
        </div>
      </section>

      <footer className="arena-footer">
        <span>All inference happens on this device.</span>
        <Link href="/">Return to the learning lab</Link>
      </footer>
    </main>
  );
}

function ModelLoader({
  label,
  role,
  slot,
  onReference,
  onLoad,
}: {
  label: string;
  role: string;
  slot: ModelSlot;
  onReference: (reference: string) => void;
  onLoad: () => void;
}) {
  const percent = slot.progress?.totalBytes
    ? Math.min(100, (slot.progress.loadedBytes / slot.progress.totalBytes) * 100)
    : null;
  return (
    <article className="model-loader">
      <header><span>{label}</span><small>{role}</small></header>
      <label htmlFor={`model-${label.toLowerCase().replace(" ", "-")}`}>Hugging Face model</label>
      <div className="model-input-row">
        <input
          id={`model-${label.toLowerCase().replace(" ", "-")}`}
          type="text"
          value={slot.reference}
          onChange={(event) => onReference(event.target.value)}
          placeholder="owner/repository@commit"
          spellCheck={false}
          autoCapitalize="none"
          autoCorrect="off"
        />
        <button type="button" onClick={onLoad} disabled={slot.phase === "loading"}>
          {slot.phase === "loading" ? "Loading…" : slot.phase === "ready" ? "Reload" : "Load"}
        </button>
      </div>
      {slot.phase === "loading" && slot.progress ? (
        <div className="load-progress" aria-live="polite">
          <div><i style={{ width: percent === null ? "24%" : `${percent}%` }} /></div>
          <span>{slot.progress.stage} · {formatBytes(slot.progress.loadedBytes)}{slot.progress.totalBytes ? ` / ${formatBytes(slot.progress.totalBytes)}` : ""}</span>
        </div>
      ) : null}
      {slot.model ? (
        <dl className="model-facts">
          <div><dt>Name</dt><dd>{slot.model.info.name}</dd></div>
          <div><dt>Runtime</dt><dd>{slot.model.info.runtime}</dd></div>
          <div><dt>Artifact</dt><dd>{formatBytes(slot.model.info.artifactBytes)}</dd></div>
          <div><dt>Revision</dt><dd className={slot.model.info.pinned ? "verified" : "mutable"}>{slot.model.info.pinned ? "Pinned" : "Mutable"}</dd></div>
          <div><dt>SHA-256</dt><dd><code>{slot.model.info.digest.slice(0, 12)}…</code></dd></div>
        </dl>
      ) : null}
      {slot.error ? <p className="model-error" role="alert">{slot.error}</p> : null}
    </article>
  );
}

function describeGame(game: Chess, running: boolean, thinking: string | null): string {
  if (game.isCheckmate()) return `${game.turn() === "w" ? "Black" : "White"} wins by checkmate`;
  if (game.isDraw()) return "Draw";
  if (game.isGameOver()) return "Game over";
  const side = game.turn() === "w" ? "White" : "Black";
  if (thinking) return `${side} to move · calculating`;
  return `${side} to move${game.isCheck() ? " · check" : ""}${running ? "" : " · paused"}`;
}

function pieceName(piece: PieceSymbol): string {
  return { p: "pawn", n: "knight", b: "bishop", r: "rook", q: "queen", k: "king" }[piece];
}

function formatBytes(bytes: number): string {
  if (bytes < 1_000_000) return `${(bytes / 1_000).toFixed(bytes < 10_000 ? 1 : 0)} kB`;
  return `${(bytes / 1_000_000).toFixed(1)} MB`;
}
