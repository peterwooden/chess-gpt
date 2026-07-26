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
const MODEL_AUTOPLAY_DELAY_MS = 650;
const FILES = ["a", "b", "c", "d", "e", "f", "g", "h"] as const;
const RANKS = [8, 7, 6, 5, 4, 3, 2, 1] as const;
const PIECES: Record<Color, Record<PieceSymbol, string>> = {
  w: { p: "♟", n: "♞", b: "♝", r: "♜", q: "♛", k: "♚" },
  b: { p: "♟", n: "♞", b: "♝", r: "♜", q: "♛", k: "♚" },
};
const CAPTURE_VALUES: Record<PieceSymbol, number> = { p: 1, n: 3, b: 3, r: 5, q: 9, k: 0 };
const CAPTURE_ORDER: PieceSymbol[] = ["q", "r", "b", "n", "p"];

type ModelSlot = {
  reference: string;
  phase: "idle" | "loading" | "ready" | "error";
  progress: LoadProgress | null;
  model: BrowserChessModel | null;
  error: string | null;
};

type PlayMode = "human" | "models";
type SidePreference = Color | "random";

type MoveRecord = {
  ply: number;
  san: string;
  actor: string;
  source: string;
  elapsedMs: number | null;
  from: Square;
  to: Square;
  color: Color;
  captured?: PieceSymbol;
};

type PromotionChoice = {
  from: Square;
  to: Square;
  pieces: PieceSymbol[];
};

type PlayerStripProps = {
  color: Color;
  name: string;
  captured: PieceSymbol[];
  lead: number;
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
  const [player1Color, setPlayer1Color] = useState<Color>("w");
  const [sidePreference, setSidePreference] = useState<SidePreference>("random");
  const [running, setRunning] = useState(false);
  const [thinking, setThinking] = useState<string | null>(null);
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
  const orientation: Color = gameStarted && mode === "human" ? humanColor : "w";
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

  function beginGame() {
    if (!modelA.model && !modelB.model) {
      setGameError("Load at least one model before starting a game.");
      return;
    }
    const nextMode: PlayMode = modelA.model && modelB.model ? "models" : "human";
    const resolvedPlayer1Color = sidePreference === "random" ? randomColor() : sidePreference;
    gameEpoch.current += 1;
    gameRef.current = new Chess();
    setMode(nextMode);
    setPlayer1Color(resolvedPlayer1Color);
    setHumanColor(modelA.model ? oppositeColor(resolvedPlayer1Color) : resolvedPlayer1Color);
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
            color: move.color,
            captured: move.captured,
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
      return game.turn() === player1Color
        ? modelA.model && { model: modelA.model, actor: modelA.model.info.name }
        : modelB.model && { model: modelB.model, actor: modelB.model.info.name };
    }
    if (game.turn() === humanColor) return null;
    const singleModel = modelA.model ?? modelB.model;
    return singleModel && { model: singleModel, actor: singleModel.info.name };
  })();

  useEffect(() => {
    if (!activeModel || thinking) return;
    const timer = window.setTimeout(
      () => void playModelMove(activeModel.model, activeModel.actor),
      mode === "models" ? MODEL_AUTOPLAY_DELAY_MS : 180,
    );
    return () => window.clearTimeout(timer);
  }, [activeModel, mode, playModelMove, thinking]);

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
          color: move.color,
          captured: move.captured,
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
    const humanOpponent = modelA.model ?? modelB.model;
    const candidate =
      mode === "models"
        ? game.turn() === player1Color
          ? modelA.model && { model: modelA.model, actor: modelA.model.info.name }
          : modelB.model && { model: modelB.model, actor: modelB.model.info.name }
        : game.turn() !== humanColor
          ? humanOpponent && { model: humanOpponent, actor: humanOpponent.info.name }
          : null;
    if (candidate) await playModelMove(candidate.model, candidate.actor);
  }

  function returnToSetup() {
    gameEpoch.current += 1;
    gameRef.current = new Chess();
    setRunning(false);
    setThinking(null);
    setMoves([]);
    setSelectedSquare(null);
    setPromotion(null);
    setGameError(null);
    setGameStarted(false);
    setGameVersion((value) => value + 1);
  }

  const player1Name = modelA.model?.info.name ?? "Human";
  const player2Name = modelB.model?.info.name ?? "Human";
  const whitePlayer = player1Color === "w" ? player1Name : player2Name;
  const blackPlayer = player1Color === "b" ? player1Name : player2Name;
  const whiteCapturedPieces = capturedPieces(moves, "w");
  const blackCapturedPieces = capturedPieces(moves, "b");
  const whiteCapturePoints = capturePoints(whiteCapturedPieces);
  const blackCapturePoints = capturePoints(blackCapturedPieces);
  const materialLead = whiteCapturePoints - blackCapturePoints;
  const whitePlayerSummary: PlayerStripProps = {
    color: "w",
    name: whitePlayer,
    captured: whiteCapturedPieces,
    lead: Math.max(0, materialLead),
  };
  const blackPlayerSummary: PlayerStripProps = {
    color: "b",
    name: blackPlayer,
    captured: blackCapturedPieces,
    lead: Math.max(0, -materialLead),
  };
  const topPlayerSummary = orientation === "w" ? blackPlayerSummary : whitePlayerSummary;
  const bottomPlayerSummary = orientation === "w" ? whitePlayerSummary : blackPlayerSummary;

  return (
    <main className="arena-page arena-page-v2">
      <nav className="arena-nav" aria-label="Arena navigation">
        <Link href="/" className="arena-title">ChessGPT arena</Link>
        <span>Runs locally in your browser</span>
      </nav>

      <section className="arena-workspace" aria-label="Chess arena">
        <div className="board-stage">
          <div className="game-status" aria-live="polite">
            <span>{gameStarted ? mode === "human" ? "Human match" : "Model match" : "Board ready"}</span>
            <strong>{gameStarted ? status : "Choose players and press Start"}</strong>
            <small>{gameStarted ? `${history.length} plies · ${Math.ceil(history.length / 2)} moves` : "Local inference"}</small>
          </div>
          <div className={`board-frame${gameStarted ? " with-players" : ""}`}>
            {gameStarted ? <PlayerStrip {...topPlayerSummary} /> : null}
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
            {gameStarted ? <PlayerStrip {...bottomPlayerSummary} /> : null}
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

        <aside className="arena-sidebar">
          {!gameStarted ? (
            <section className="setup-pane" aria-labelledby="setup-title">
              <header className="setup-heading">
                <p className="eyebrow">New game</p>
                <h1 id="setup-title">Set up game</h1>
                <p>Load one model to play it yourself, or load two to watch them play.</p>
              </header>

              <ModelLoader
                label="Player 1"
                role="Model · optional"
                slot={modelA}
                onReference={(reference) => updateReference("a", reference)}
                onLoad={() => void loadModel("a")}
              />

              <fieldset className="side-picker">
                <legend>Player 1 plays</legend>
                <div>
                  {([
                    { value: "w", label: "White" },
                    { value: "random", label: "Random" },
                    { value: "b", label: "Black" },
                  ] as const).map((side) => (
                    <button
                      type="button"
                      className={sidePreference === side.value ? "selected" : ""}
                      aria-pressed={sidePreference === side.value}
                      onClick={() => setSidePreference(side.value)}
                      key={side.value}
                    >
                      {side.label}
                    </button>
                  ))}
                </div>
              </fieldset>

              <ModelLoader
                label="Player 2"
                role="Optional · Human if empty"
                slot={modelB}
                onReference={(reference) => updateReference("b", reference)}
                onLoad={() => void loadModel("b")}
              />

              <div className="setup-actions">
                {gameError ? <p className="arena-error" role="alert">{gameError}</p> : null}
                <p>{modelA.model && modelB.model ? "Model versus model" : modelA.model || modelB.model ? "The empty player slot will be Human" : "Load at least one model"}</p>
                <button type="button" onClick={beginGame} disabled={(!modelA.model && !modelB.model) || modelA.phase === "loading" || modelB.phase === "loading"}>
                  Start game
                </button>
              </div>
            </section>
          ) : (
            <section className="move-console" aria-label="Game progress">
              <header>
                <div><span>Moves</span><strong>{thinking ? `${thinking} is thinking` : running ? "Game running" : "Game paused"}</strong></div>
                <i className={thinking ? "pulse active" : "pulse"} aria-hidden="true" />
              </header>
              {gameError ? <p className="arena-error" role="alert">{gameError}</p> : null}
              {moves.length === 0 ? (
                <div className="empty-record">
                  <span>01</span>
                  <p>The game is ready. The first SAN move will appear here.</p>
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
              <div className="game-controls">
                <button
                  type="button"
                  onClick={() => setRunning((value) => !value)}
                  disabled={game.isGameOver()}
                >
                  {running ? "Pause" : "Resume"}
                </button>
                <button
                  type="button"
                  onClick={() => void stepOnce()}
                  disabled={running || Boolean(thinking) || game.isGameOver() || (mode === "human" && game.turn() === humanColor)}
                >
                  Next move
                </button>
                <button type="button" onClick={returnToSetup}>New game</button>
              </div>
            </section>
          )}
        </aside>
      </section>
    </main>
  );
}

function PlayerStrip({ color, name, captured, lead }: PlayerStripProps) {
  const colorName = color === "w" ? "White" : "Black";
  const capturedColor = oppositeColor(color);

  return (
    <section className={`player-strip ${color === "w" ? "white" : "black"}`} aria-label={`${colorName} player`}>
      <div className="player-identity">
        <span>{colorName}</span>
        <strong>{name}</strong>
      </div>
      <div className="captured-pieces" aria-label={`${colorName} captured pieces`}>
        {captured.length > 0 ? captured.map((piece, index) => (
          <span
            className={`captured-piece ${capturedColor === "w" ? "white" : "black"}`}
            aria-label={`Captured ${capturedColor === "w" ? "white" : "black"} ${pieceName(piece)}`}
            key={`${piece}-${index}`}
          >
            {PIECES[capturedColor][piece]}
          </span>
        )) : <span className="no-captures" aria-hidden="true">—</span>}
        {lead > 0 ? <strong className="material-lead">+{lead}</strong> : null}
      </div>
    </section>
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
        <div className="model-summary">
          <strong>{slot.model.info.name}</strong>
          <span>{slot.model.info.runtime} · {formatBytes(slot.model.info.artifactBytes)} · {slot.model.info.pinned ? "Pinned" : "Mutable"}</span>
          <code title={`SHA-256 ${slot.model.info.digest}`}>SHA {slot.model.info.digest.slice(0, 12)}…</code>
        </div>
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

function oppositeColor(color: Color): Color {
  return color === "w" ? "b" : "w";
}

function randomColor(): Color {
  return crypto.getRandomValues(new Uint8Array(1))[0] % 2 === 0 ? "w" : "b";
}

function capturedPieces(moves: MoveRecord[], color: Color): PieceSymbol[] {
  return moves
    .flatMap((move) => move.color === color && move.captured ? [move.captured] : [])
    .sort((left, right) => CAPTURE_ORDER.indexOf(left) - CAPTURE_ORDER.indexOf(right));
}

function capturePoints(pieces: PieceSymbol[]): number {
  return pieces.reduce((total, piece) => total + CAPTURE_VALUES[piece], 0);
}

function pieceName(piece: PieceSymbol): string {
  return { p: "pawn", n: "knight", b: "bishop", r: "rook", q: "queen", k: "king" }[piece];
}

function formatBytes(bytes: number): string {
  if (bytes < 1_000_000) return `${(bytes / 1_000).toFixed(bytes < 10_000 ? 1 : 0)} kB`;
  return `${(bytes / 1_000_000).toFixed(1)} MB`;
}
