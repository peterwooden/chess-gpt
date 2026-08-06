"use client";

import { Chess, type Color, type PieceSymbol, type Square } from "chess.js";
import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  DEFAULT_MOVE_TIME_LIMIT_MS,
  loadBrowserModel,
  type BrowserChessModel,
  type LoadProgress,
} from "./model";
import { modelPageHref } from "./hugging-face-reference.mjs";
import {
  buildArenaShareUrl,
  readSharedModelReferences,
  readSharedPgn,
  withSharedModelReference,
} from "./share-url.mjs";
import {
  analyzeGameWithStockfish,
  type GameReview,
  type MoveReview,
  type PlayerReview,
  type ReviewJudgement,
  type ReviewProgress,
} from "./stockfish-review.mjs";

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
const POSITIVE_JUDGEMENTS = ["brilliant", "good"] as const;
const NEGATIVE_JUDGEMENTS = ["inaccuracy", "mistake", "blunder"] as const;
const JUDGEMENT_META: Record<ReviewJudgement, { glyph: string; label: string; plural: string }> = {
  brilliant: { glyph: "!!", label: "Brilliant", plural: "brilliant moves" },
  good: { glyph: "!", label: "Good", plural: "good moves" },
  inaccuracy: { glyph: "?!", label: "Inaccuracy", plural: "inaccuracies" },
  mistake: { glyph: "?", label: "Mistake", plural: "mistakes" },
  blunder: { glyph: "??", label: "Blunder", plural: "blunders" },
};

type ModelSlot = {
  reference: string;
  phase: "idle" | "loading" | "ready" | "error";
  progress: LoadProgress | null;
  model: BrowserChessModel | null;
  profileId: string | null;
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
  profileId?: string | null;
  modelReference?: string | null;
  captured: PieceSymbol[];
  lead: number;
};

type Players = Record<Color, string>;
type PlayerProfiles = Record<Color, string | null>;
type PlayerModelReferences = Record<Color, string | null>;

type StoredGame = {
  id: string;
  whitePlayerId: string | null;
  blackPlayerId: string | null;
  whiteName: string;
  blackName: string;
  whiteModelReference: string | null;
  blackModelReference: string | null;
  pgn: string;
};

type SaveState = {
  phase: "idle" | "saving" | "saved" | "error";
  gameId: string | null;
  message: string;
};

type ReviewState = {
  phase: "idle" | "loading" | "analyzing" | "complete" | "error";
  progress: ReviewProgress | null;
  result: GameReview | null;
  error: string | null;
};

function emptySlot(): ModelSlot {
  return { reference: "", phase: "idle", progress: null, model: null, profileId: null, error: null };
}

function emptyReview(): ReviewState {
  return { phase: "idle", progress: null, result: null, error: null };
}

export default function ArenaClient({ viewer }: { viewer: { signedIn: boolean; name: string | null; profileId: string | null } }) {
  const gameRef = useRef(new Chess());
  const moveRecordRef = useRef<HTMLOListElement>(null);
  const gameEpoch = useRef(0);
  const reviewAbort = useRef<AbortController | null>(null);
  const loadEpoch = useRef({ a: 0, b: 0 });
  const activeGameId = useRef("");
  const activeGameStartedAt = useRef("");
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
  const [moveTimeLimitMsA, setMoveTimeLimitMsA] = useState(DEFAULT_MOVE_TIME_LIMIT_MS);
  const [moveTimeLimitMsB, setMoveTimeLimitMsB] = useState(DEFAULT_MOVE_TIME_LIMIT_MS);
  const [running, setRunning] = useState(false);
  const [starting, setStarting] = useState(false);
  const [thinking, setThinking] = useState<string | null>(null);
  const [gameStarted, setGameStarted] = useState(false);
  const [players, setPlayers] = useState<Players>({ w: "White", b: "Black" });
  const [playerProfiles, setPlayerProfiles] = useState<PlayerProfiles>({ w: null, b: null });
  const [playerModelReferences, setPlayerModelReferences] = useState<PlayerModelReferences>({ w: null, b: null });
  const [recordingEnabled, setRecordingEnabled] = useState(false);
  const [saveState, setSaveState] = useState<SaveState>({ phase: "idle", gameId: null, message: "" });
  const [finishedStatus, setFinishedStatus] = useState<string | null>(null);
  const [shareOpen, setShareOpen] = useState(false);
  const [shareMessage, setShareMessage] = useState("");
  const [, setGameVersion] = useState(0);
  const [moves, setMoves] = useState<MoveRecord[]>([]);
  const [viewedPly, setViewedPly] = useState<number | null>(null);
  const [historyPlaying, setHistoryPlaying] = useState(false);
  const [selectedSquare, setSelectedSquare] = useState<Square | null>(null);
  const [promotion, setPromotion] = useState<PromotionChoice | null>(null);
  const [gameError, setGameError] = useState<string | null>(null);
  const [review, setReview] = useState<ReviewState>(emptyReview);
  const [reviewAttempt, setReviewAttempt] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    const restorePgn = (pgn: string, stored?: StoredGame) => {
      const restored = new Chess();
      restored.loadPgn(pgn);
      const headers = restored.getHeaders();
      gameRef.current = restored;
      const white = stored?.whiteName ?? headers.White ?? "White";
      const black = stored?.blackName ?? headers.Black ?? "Black";
      setMoves(recordsFromGame(restored, white, black));
      setPlayers({ w: white, b: black });
      setPlayerProfiles({
        w: stored?.whitePlayerId ?? null,
        b: stored?.blackPlayerId ?? null,
      });
      setPlayerModelReferences({
        w: stored?.whiteModelReference ?? null,
        b: stored?.blackModelReference ?? null,
      });
      const modelReferences = [stored?.whiteModelReference, stored?.blackModelReference].filter(
        (value): value is string => Boolean(value),
      );
      if (stored) {
        setModelA((current) => ({ ...current, reference: modelReferences[0] ?? "" }));
        setModelB((current) => ({ ...current, reference: modelReferences[1] ?? "" }));
        setMode(modelReferences.length === 2 ? "models" : "human");
        const firstModelColor: Color = stored.whiteModelReference ? "w" : "b";
        setPlayer1Color(firstModelColor);
        setHumanColor(firstModelColor === "w" ? "b" : "w");
        setSaveState({ phase: "saved", gameId: stored.id, message: "Saved to history." });
      } else {
        setMode(headers.Mode === "models" ? "models" : "human");
        setPlayer1Color(headers.PlayerOneColor === "Black" ? "b" : "w");
        setHumanColor(headers.HumanColor === "Black" ? "b" : "w");
      }
      setRecordingEnabled(false);
      setFinishedStatus(describePgnResult(restored));
      setGameStarted(true);
      setRunning(false);
      setGameVersion((value) => value + 1);
    };

    const url = new URL(window.location.href);
    const storedGameId = url.searchParams.get("game")?.trim();
    if (storedGameId) {
      void fetch(`/api/games/${encodeURIComponent(storedGameId)}`, { signal: controller.signal })
        .then(async (response) => {
          if (!response.ok) throw new Error("This recorded game could not be loaded.");
          return response.json() as Promise<{ game: StoredGame }>;
        })
        .then(({ game }) => restorePgn(game.pgn, game))
        .catch((error) => {
          if (error instanceof DOMException && error.name === "AbortError") return;
          setGameError(error instanceof Error ? error.message : "This recorded game could not be loaded.");
        });
      return () => controller.abort();
    }

    const shared = readSharedModelReferences(window.location.href);
    if (shared) {
      setModelA((current) => ({ ...current, reference: shared.a }));
      setModelB((current) => ({ ...current, reference: shared.b }));
    }
    if (!shared) {
      const saved = window.localStorage.getItem(MODEL_URLS_KEY);
      if (saved) {
        try {
          const urls = JSON.parse(saved) as { a?: string; b?: string };
          setModelA((current) => ({ ...current, reference: urls.a ?? "" }));
          setModelB((current) => ({ ...current, reference: urls.b ?? "" }));
        } catch {
          window.localStorage.removeItem(MODEL_URLS_KEY);
        }
      }
    }

    const sharedPgn = readSharedPgn(window.location.href);
    if (sharedPgn) {
      try {
        restorePgn(sharedPgn);
        if (shared) {
          const headers = gameRef.current.getHeaders();
          const firstColor: Color = headers.PlayerOneColor === "Black" ? "b" : "w";
          setPlayerModelReferences({
            [firstColor]: shared.a || null,
            [oppositeColor(firstColor)]: shared.b || null,
          } as PlayerModelReferences);
        }
      } catch {
        setGameError("This shared PGN could not be loaded.");
      }
    }
    return () => controller.abort();
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

  useEffect(() => {
    const record = moveRecordRef.current;
    if (!record) return;
    if (viewedPly === null) {
      record.scrollTop = record.scrollHeight;
      return;
    }
    if (viewedPly === 0) {
      record.scrollTop = 0;
      return;
    }
    const selectedMove = record.querySelector<HTMLElement>(`[data-ply="${viewedPly}"]`);
    const selectedRow = selectedMove?.closest("li");
    if (!selectedRow) return;
    const rowTop = selectedRow.offsetTop;
    const rowBottom = rowTop + selectedRow.offsetHeight;
    if (rowTop < record.scrollTop) record.scrollTop = rowTop;
    else if (rowBottom > record.scrollTop + record.clientHeight) {
      record.scrollTop = rowBottom - record.clientHeight;
    }
  }, [moves.length, viewedPly]);

  useEffect(() => {
    if (!historyPlaying) return;
    if (moves.length === 0) {
      setHistoryPlaying(false);
      return;
    }
    const timer = window.setTimeout(() => {
      if (viewedPly === null || viewedPly >= moves.length - 1) {
        setViewedPly(null);
        setHistoryPlaying(false);
      } else {
        setViewedPly(viewedPly + 1);
      }
    }, 700);
    return () => window.clearTimeout(timer);
  }, [historyPlaying, moves.length, viewedPly]);

  useEffect(() => () => {
    loadEpoch.current.a += 1;
    loadEpoch.current.b += 1;
    reviewAbort.current?.abort();
    void loadedModels.current.a?.dispose();
    void loadedModels.current.b?.dispose();
  }, []);

  const game = gameRef.current;
  const history = game.history();
  const isLiveView = viewedPly === null;
  const displayPly = viewedPly ?? moves.length;
  const displayedGame = isLiveView ? game : gameAtPly(moves, displayPly);
  const targetSquares = isLiveView && selectedSquare
    ? new Set(game.moves({ square: selectedSquare, verbose: true }).map((move) => move.to))
    : new Set<Square>();
  const displayedMove = displayPly > 0 ? moves[displayPly - 1] : undefined;
  const orientation: Color = gameStarted && mode === "human" ? humanColor : "w";
  const boardRanks = orientation === "w" ? RANKS : [...RANKS].reverse();
  const boardFiles = orientation === "w" ? FILES : [...FILES].reverse();
  const status = finishedStatus ?? describeGame(game, running, thinking);
  const displayedStatus = isLiveView
    ? status
    : describeHistoryPosition(displayPly, displayedMove);
  const gameEnded = game.isGameOver() || finishedStatus !== null;
  const moveRows = pairMoves(moves);
  const reviewByPly = new Map(review.result?.moves.map((move) => [move.ply, move]) ?? []);

  useEffect(() => {
    if (!recordingEnabled || !gameStarted || !gameEnded || moves.length === 0 || saveState.phase !== "idle") return;
    const participantFor = (color: Color) => {
      const slot = color === player1Color ? modelA : modelB;
      return slot.model
        ? { kind: "model" as const, reference: slot.model.info.reference }
        : { kind: "human" as const };
    };
    setSaveState({ phase: "saving", gameId: null, message: "Saving game…" });
    const pgn = createSharePgn(game, players, mode, player1Color, humanColor, gameError);
    void fetch("/api/games", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        id: activeGameId.current,
        pgn,
        playedAt: activeGameStartedAt.current,
        white: participantFor("w"),
        black: participantFor("b"),
      }),
    }).then(async (response) => {
      const payload = await response.json() as { game?: StoredGame; error?: string };
      if (!response.ok || !payload.game) throw new Error(payload.error ?? "The game could not be saved.");
      setPlayers({ w: payload.game.whiteName, b: payload.game.blackName });
      setPlayerProfiles({ w: payload.game.whitePlayerId, b: payload.game.blackPlayerId });
      setSaveState({ phase: "saved", gameId: payload.game.id, message: "Saved to history." });
    }).catch((error) => {
      setSaveState({
        phase: "error",
        gameId: null,
        message: error instanceof Error ? error.message : "The game could not be saved.",
      });
    });
  }, [
    gameEnded,
    gameError,
    game,
    gameStarted,
    humanColor,
    mode,
    modelA,
    modelB,
    moves.length,
    player1Color,
    players,
    recordingEnabled,
    saveState.phase,
  ]);

  useEffect(() => {
    if (!gameStarted || !gameEnded || moves.length === 0) return;
    reviewAbort.current?.abort();
    const controller = new AbortController();
    reviewAbort.current = controller;
    setReview({ phase: "loading", progress: null, result: null, error: null });
    void analyzeGameWithStockfish(
      moves.map((move) => move.san),
      (progress) => {
        if (controller.signal.aborted) return;
        setReview((current) => ({ ...current, phase: "analyzing", progress }));
      },
      controller.signal,
    ).then((result) => {
      if (controller.signal.aborted) return;
      setReview({ phase: "complete", progress: null, result, error: null });
    }).catch((error) => {
      if (controller.signal.aborted || (error instanceof DOMException && error.name === "AbortError")) return;
      setReview({
        phase: "error",
        progress: null,
        result: null,
        error: error instanceof Error ? error.message : "The game could not be analysed.",
      });
    });
    return () => controller.abort();
  }, [gameEnded, gameStarted, moves, reviewAttempt]);

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
      setSlot((value) => ({ ...value, phase: "loading", progress: null, model: null, profileId: null, error: null }));
      let loaded: BrowserChessModel | null = null;
      try {
        loaded = await loadBrowserModel(current.reference, (progress) => {
          if (loadEpoch.current[slot] !== requestId) return;
          setSlot((value) => ({ ...value, progress }));
        });
        if (loadEpoch.current[slot] !== requestId) {
          await loaded.dispose();
          return;
        }
        const profile = await resolveModelProfile(loaded.info.reference);
        if (loadEpoch.current[slot] !== requestId) {
          await loaded.dispose();
          return;
        }
        const shareUrl = withSharedModelReference(
          window.location.href,
          slot,
          loaded.info.reference,
        );
        window.history.replaceState(window.history.state, "", shareUrl);
        setSlot((value) => ({
          ...value,
          reference: loaded.info.reference,
          phase: "ready",
          progress: null,
          model: loaded,
          profileId: profile.id,
          error: null,
        }));
      } catch (error) {
        await loaded?.dispose();
        if (loadEpoch.current[slot] !== requestId) return;
        setSlot((value) => ({
          ...value,
          phase: "error",
          progress: null,
          model: null,
          profileId: null,
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

  async function beginGame() {
    if (!modelA.model && !modelB.model) {
      setGameError("Load at least one model before starting a game.");
      return;
    }
    const nextMode: PlayMode = modelA.model && modelB.model ? "models" : "human";
    const resolvedPlayer1Color = sidePreference === "random" ? randomColor() : sidePreference;
    const humanName = viewer.signedIn ? viewer.name ?? "Signed-in player" : "Anonymous";
    const nextPlayer1Name = modelA.model?.info.name ?? humanName;
    const nextPlayer2Name = modelB.model?.info.name ?? humanName;
    const seed = crypto.getRandomValues(new Uint32Array(1))[0];
    setStarting(true);
    setGameError(null);
    const startingModels = [
      modelA.model && { model: modelA.model, seed },
      modelB.model && { model: modelB.model, seed: seed ^ 0x9e3779b9 },
    ].filter((item): item is { model: BrowserChessModel; seed: number } => Boolean(item));
    const startResults = await Promise.allSettled(
      startingModels.map(({ model, seed: modelSeed }) => model.newGame(modelSeed)),
    );
    const failedIndex = startResults.findIndex((result) => result.status === "rejected");
    if (failedIndex >= 0) {
      const result = startResults[failedIndex] as PromiseRejectedResult;
      const detail = result.reason instanceof Error ? result.reason.message : "failed to start";
      setGameError(`${startingModels[failedIndex].model.info.name} loses before move 1: ${detail}`);
      setStarting(false);
      return;
    }
    gameEpoch.current += 1;
    reviewAbort.current?.abort();
    gameRef.current = new Chess();
    activeGameId.current = crypto.randomUUID();
    activeGameStartedAt.current = new Date().toISOString();
    setMode(nextMode);
    setPlayer1Color(resolvedPlayer1Color);
    setHumanColor(modelA.model ? oppositeColor(resolvedPlayer1Color) : resolvedPlayer1Color);
    setPlayers({
      [resolvedPlayer1Color]: nextPlayer1Name,
      [oppositeColor(resolvedPlayer1Color)]: nextPlayer2Name,
    } as Players);
    const nextPlayer1Profile = modelA.model ? modelA.profileId : viewer.profileId;
    const nextPlayer2Profile = modelB.model ? modelB.profileId : viewer.profileId;
    setPlayerProfiles({
      [resolvedPlayer1Color]: nextPlayer1Profile,
      [oppositeColor(resolvedPlayer1Color)]: nextPlayer2Profile,
    } as PlayerProfiles);
    setPlayerModelReferences({
      [resolvedPlayer1Color]: modelA.model?.info.reference ?? null,
      [oppositeColor(resolvedPlayer1Color)]: modelB.model?.info.reference ?? null,
    } as PlayerModelReferences);
    setRecordingEnabled(true);
    setSaveState({ phase: "idle", gameId: null, message: "" });
    setMoves([]);
    setViewedPly(null);
    setHistoryPlaying(false);
    setSelectedSquare(null);
    setPromotion(null);
    setThinking(null);
    setGameError(null);
    setFinishedStatus(null);
    setShareOpen(false);
    setShareMessage("");
    setReview(emptyReview());
    setGameStarted(true);
    setRunning(true);
    setStarting(false);
    setGameVersion((value) => value + 1);
  }

  const playModelMove = useCallback(
    async (model: BrowserChessModel, actor: string, moveTimeLimitMs: number) => {
      const epoch = gameEpoch.current;
      const activeGame = gameRef.current;
      if (activeGame.isGameOver()) return;
      setThinking(actor);
      setGameError(null);
      const started = performance.now();
      try {
        const prediction = await model.predict(activeGame.history(), activeGame.moves(), moveTimeLimitMs);
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
        const detail = error instanceof Error ? error.message : "failed to return a legal move";
        setFinishedStatus(`${activeGame.turn() === "w" ? "Black" : "White"} wins by forfeit`);
        setGameError(`${actor} loses: ${detail}`);
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
        ? modelA.model && {
            model: modelA.model,
            actor: modelA.model.info.name,
            moveTimeLimitMs: moveTimeLimitMsA,
          }
        : modelB.model && {
            model: modelB.model,
            actor: modelB.model.info.name,
            moveTimeLimitMs: moveTimeLimitMsB,
          };
    }
    if (game.turn() === humanColor) return null;
    const singleModel = modelA.model ?? modelB.model;
    return singleModel && {
      model: singleModel,
      actor: singleModel.info.name,
      moveTimeLimitMs: modelA.model ? moveTimeLimitMsA : moveTimeLimitMsB,
    };
  })();

  useEffect(() => {
    if (!activeModel || thinking) return;
    const timer = window.setTimeout(
      () => void playModelMove(
        activeModel.model,
        activeModel.actor,
        activeModel.moveTimeLimitMs,
      ),
      mode === "models" ? MODEL_AUTOPLAY_DELAY_MS : 180,
    );
    return () => window.clearTimeout(timer);
  }, [activeModel, mode, playModelMove, thinking]);

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      // Don't intercept arrow keys when the user is typing in a form field
      // or editing contenteditable text.
      const target = event.target;
      if (
        target instanceof HTMLInputElement ||
        target instanceof HTMLTextAreaElement ||
        (target instanceof HTMLElement && target.isContentEditable)
      ) {
        return;
      }
      // Only navigate when there is a game to replay.
      if (moves.length === 0) return;

      // displayPly mirrors the "Next" / "Previous" button logic:
      // null (live view) resolves to moves.length (the end of the game).
      const displayPly = viewedPly ?? moves.length;

      if (event.key === "ArrowLeft") {
        if (displayPly === 0) return; // can't go before the start
        event.preventDefault();
        setHistoryPlaying(false);
        setSelectedSquare(null);
        setPromotion(null);
        setViewedPly(displayPly - 1);
      } else if (event.key === "ArrowRight") {
        if (viewedPly === null) return; // can't go forward from the live/end position
        event.preventDefault();
        setHistoryPlaying(false);
        setSelectedSquare(null);
        setPromotion(null);
        // Mirrors showPosition's guard: advancing past the last
        // move returns to the live (null) view.
        const nextPly = displayPly + 1;
        setViewedPly(nextPly >= moves.length ? null : nextPly);
      }
    }

    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [viewedPly, moves.length]);

  function chooseSquare(square: Square) {
    if (!isLiveView || mode !== "human" || !running || thinking || game.turn() !== humanColor) return;
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
    const candidate =
      mode === "models"
        ? game.turn() === player1Color
          ? modelA.model && {
              model: modelA.model,
              actor: modelA.model.info.name,
              moveTimeLimitMs: moveTimeLimitMsA,
            }
          : modelB.model && {
              model: modelB.model,
              actor: modelB.model.info.name,
              moveTimeLimitMs: moveTimeLimitMsB,
            }
        : game.turn() !== humanColor
          ? modelA.model
            ? {
                model: modelA.model,
                actor: modelA.model.info.name,
                moveTimeLimitMs: moveTimeLimitMsA,
              }
            : modelB.model && {
                model: modelB.model,
                actor: modelB.model.info.name,
                moveTimeLimitMs: moveTimeLimitMsB,
              }
          : null;
    if (candidate) {
      await playModelMove(candidate.model, candidate.actor, candidate.moveTimeLimitMs);
    }
  }

  function showPosition(ply: number | null) {
    setHistoryPlaying(false);
    setSelectedSquare(null);
    setPromotion(null);
    setViewedPly(ply === null || ply >= moves.length ? null : Math.max(0, ply));
  }

  function toggleHistoryPlayback() {
    if (historyPlaying) {
      setHistoryPlaying(false);
      return;
    }
    if (moves.length === 0) return;
    setSelectedSquare(null);
    setPromotion(null);
    setViewedPly((current) => current === null ? 0 : current);
    setHistoryPlaying(true);
  }

  function returnToSetup() {
    gameEpoch.current += 1;
    reviewAbort.current?.abort();
    gameRef.current = new Chess();
    const setupUrl = new URL(window.location.href);
    setupUrl.searchParams.delete("game");
    window.history.replaceState(
      window.history.state,
      "",
      buildArenaShareUrl(setupUrl, {
        a: modelA.reference,
        b: modelB.reference,
        pgn: null,
      }),
    );
    setRunning(false);
    setThinking(null);
    setMoves([]);
    setViewedPly(null);
    setHistoryPlaying(false);
    setSelectedSquare(null);
    setPromotion(null);
    setGameError(null);
    setFinishedStatus(null);
    setShareOpen(false);
    setShareMessage("");
    setReview(emptyReview());
    setPlayers({ w: "White", b: "Black" });
    setPlayerProfiles({ w: null, b: null });
    setPlayerModelReferences({ w: null, b: null });
    setRecordingEnabled(false);
    setSaveState({ phase: "idle", gameId: null, message: "" });
    setGameStarted(false);
    setGameVersion((value) => value + 1);
  }

  const displayedMoves = moves.slice(0, displayPly);
  const whiteCapturedPieces = capturedPieces(displayedMoves, "w");
  const blackCapturedPieces = capturedPieces(displayedMoves, "b");
  const whiteCapturePoints = capturePoints(whiteCapturedPieces);
  const blackCapturePoints = capturePoints(blackCapturedPieces);
  const materialLead = whiteCapturePoints - blackCapturePoints;
  const whitePlayerSummary: PlayerStripProps = {
    color: "w",
    name: players.w,
    profileId: playerProfiles.w,
    modelReference: playerModelReferences.w,
    captured: whiteCapturedPieces,
    lead: Math.max(0, materialLead),
  };
  const blackPlayerSummary: PlayerStripProps = {
    color: "b",
    name: players.b,
    profileId: playerProfiles.b,
    modelReference: playerModelReferences.b,
    captured: blackCapturedPieces,
    lead: Math.max(0, -materialLead),
  };
  const topPlayerSummary = orientation === "w" ? blackPlayerSummary : whitePlayerSummary;
  const bottomPlayerSummary = orientation === "w" ? whitePlayerSummary : blackPlayerSummary;

  async function shareGame(includePgn: boolean) {
    const base = new URL(window.location.href);
    base.searchParams.delete("game");
    const url = includePgn && saveState.gameId
      ? `${window.location.origin}/arena?game=${encodeURIComponent(saveState.gameId)}`
      : buildArenaShareUrl(base, {
          a: modelA.reference,
          b: modelB.reference,
          pgn: includePgn
            ? createSharePgn(game, players, mode, player1Color, humanColor, gameError)
            : null,
        });
    setShareOpen(false);
    try {
      if (navigator.share) {
        await navigator.share({
          title: includePgn ? "ChessGPT arena game" : "ChessGPT arena models",
          url,
        });
        setShareMessage("Shared.");
      } else {
        await navigator.clipboard.writeText(url);
        setShareMessage(includePgn ? "Game link copied." : "Model link copied.");
      }
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") return;
      setShareMessage("Could not share this link.");
    }
  }

  return (
    <main className="arena-page arena-page-v2">
      <nav className="arena-nav" aria-label="Arena navigation">
        <Link href="/" className="arena-title">ChessGPT arena</Link>
        <div className="arena-nav-links"><Link href="/models">Models</Link><Link href="/history">Players</Link><Link href="/tournaments">Tournaments</Link></div>
      </nav>

      <section className={`arena-workspace ${gameStarted ? "game-mode" : "setup-mode"}`} aria-label="Chess arena">
        <div className="board-stage">
          <div className={`board-frame${gameStarted ? " with-players" : ""}`}>
            {gameStarted ? <PlayerStrip {...topPlayerSummary} /> : null}
            <div className={`chessboard orientation-${orientation}`} role="grid" aria-label="Chess board">
              {boardRanks.flatMap((rank) =>
                boardFiles.map((file) => {
                  const square = `${file}${rank}` as Square;
                  const piece = displayedGame.get(square);
                  const light = (FILES.indexOf(file) + rank) % 2 === 0;
                  const selected = square === selectedSquare;
                  const target = targetSquares.has(square);
                  const last = square === displayedMove?.from || square === displayedMove?.to;
                  return (
                    <button
                      type="button"
                      role="gridcell"
                      aria-label={`${square}${piece ? ` ${piece.color === "w" ? "white" : "black"} ${pieceName(piece.type)}` : " empty"}`}
                      className={`board-square ${light ? "light" : "dark"}${selected ? " selected" : ""}${target ? " target" : ""}${last ? " last" : ""}`}
                      onClick={() => chooseSquare(square)}
                      aria-disabled={!isLiveView}
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
                    className="promotion-piece"
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
                moveTimeLimitMs={moveTimeLimitMsA}
                onMoveTimeLimitMs={setMoveTimeLimitMsA}
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
                moveTimeLimitMs={moveTimeLimitMsB}
                onMoveTimeLimitMs={setMoveTimeLimitMsB}
                onReference={(reference) => updateReference("b", reference)}
                onLoad={() => void loadModel("b")}
              />

              <div className="setup-actions">
                {gameError ? <p className="arena-error" role="alert">{gameError}</p> : null}
                <p className="identity-notice">
                  {viewer.signedIn
                    ? `Human games will be recorded as ${viewer.name ?? "your signed-in player"}.`
                    : <><Link href="/signin-with-chatgpt?return_to=%2Farena">Sign in with ChatGPT</Link> to attach human games to your history.</>}
                </p>
                <p>{modelA.model && modelB.model ? "Model versus model" : modelA.model || modelB.model ? "The empty player slot will be Human" : "Load at least one model"}</p>
                <button type="button" onClick={() => void beginGame()} disabled={starting || (!modelA.model && !modelB.model) || modelA.phase === "loading" || modelB.phase === "loading"}>
                  {starting ? "Starting…" : "Start game"}
                </button>
              </div>
            </section>
          ) : (
            <section className="move-console" aria-label="Game progress">
              <header>
                <div>
                  <span>
                    {gameEnded && review.phase === "complete"
                      ? `Game review · ${review.result?.engine}`
                      : mode === "human" ? "Human match" : "Model match"}
                    {!isLiveView && moves.length > 0 ? ` · live at move ${Math.ceil(moves.length / 2)}` : ""}
                  </span>
                  <strong aria-live="polite">{displayedStatus}</strong>
                </div>
                <div className="move-header-meta">
                  {isLiveView ? (
                    <small>{history.length} plies · {Math.ceil(history.length / 2)} moves</small>
                  ) : (
                    <small className="history-view-label">History</small>
                  )}
                  <i className={thinking ? "pulse active" : "pulse"} aria-hidden="true" />
                </div>
              </header>
              <nav className="history-navigation" aria-label="Move history navigation">
                <button
                  type="button"
                  aria-label="Go to starting position"
                  onClick={() => showPosition(0)}
                  disabled={moves.length === 0 || displayPly === 0}
                >
                  <span aria-hidden="true">|‹</span>
                </button>
                <button
                  type="button"
                  aria-label="Go to previous position"
                  onClick={() => showPosition(displayPly - 1)}
                  disabled={moves.length === 0 || displayPly === 0}
                >
                  <span aria-hidden="true">‹</span>
                </button>
                <button
                  className="history-playback"
                  type="button"
                  aria-label={historyPlaying ? "Pause move history" : "Play move history"}
                  aria-pressed={historyPlaying}
                  onClick={toggleHistoryPlayback}
                  disabled={moves.length === 0}
                >
                  <span aria-hidden="true">{historyPlaying ? "Ⅱ" : "▶"}</span>
                </button>
                <button
                  type="button"
                  aria-label="Go to next position"
                  onClick={() => showPosition(displayPly + 1)}
                  disabled={isLiveView || moves.length === 0}
                >
                  <span aria-hidden="true">›</span>
                </button>
                <button
                  type="button"
                  aria-label="Go to live position"
                  onClick={() => showPosition(null)}
                  disabled={isLiveView || moves.length === 0}
                >
                  <span aria-hidden="true">›|</span>
                </button>
              </nav>
              {gameEnded && review.phase !== "idle" ? (
                <GameReviewSummary
                  review={review}
                  players={players}
                  profiles={playerProfiles}
                  modelReferences={playerModelReferences}
                  onRetry={() => setReviewAttempt((attempt) => attempt + 1)}
                />
              ) : null}
              {gameError ? <p className="arena-error" role="alert">{gameError}</p> : null}
              {moves.length === 0 ? (
                <div className="empty-record">
                  <span>01</span>
                  <p>The game is ready. The first SAN move will appear here.</p>
                </div>
              ) : (
                <div className="move-score">
                  <div className="move-score-heading" aria-hidden="true">
                    <span>Move</span><b>White</b><b>Black</b>
                  </div>
                  <ol className="move-record" aria-label="Move history" ref={moveRecordRef}>
                    {moveRows.map((row) => (
                      <li key={row.number}>
                        <span>{row.number}.</span>
                        <MoveCell move={row.white} review={row.white ? reviewByPly.get(row.white.ply) : undefined} currentPly={displayPly} onSelect={showPosition} />
                        <MoveCell move={row.black} review={row.black ? reviewByPly.get(row.black.ply) : undefined} currentPly={displayPly} onSelect={showPosition} />
                      </li>
                    ))}
                  </ol>
                </div>
              )}
              {gameEnded && moves.length > 0 ? (
                <PgnExport pgn={createSharePgn(game, players, mode, player1Color, humanColor, gameError)} />
              ) : null}
              {gameEnded && shareMessage ? <p className="share-feedback" role="status">{shareMessage}</p> : null}
              {gameEnded && saveState.message ? (
                <p className={`save-feedback ${saveState.phase}`} role="status">
                  <span>{saveState.message}</span>
                  {saveState.phase === "saved" && saveState.gameId ? <Link href={`/arena?game=${saveState.gameId}`}>Recorded game</Link> : null}
                  {saveState.phase === "error" ? <button type="button" onClick={() => setSaveState({ phase: "idle", gameId: null, message: "" })}>Retry</button> : null}
                </p>
              ) : null}
              {gameEnded && shareOpen ? (
                <div className="share-options" id="arena-share-options" role="menu" aria-label="Share game">
                  <button type="button" role="menuitem" onClick={() => void shareGame(false)}>
                    <strong>Models only</strong>
                    <span>Start a fresh game with these model references.</span>
                  </button>
                  <button type="button" role="menuitem" onClick={() => void shareGame(true)}>
                    <strong>Models + game</strong>
                    <span>Open this completed game and its move history.</span>
                  </button>
                </div>
              ) : null}
              <div className={`game-controls${gameEnded ? " game-ended" : ""}`}>
                {gameEnded ? (
                  <button
                    className="share-toggle"
                    type="button"
                    aria-expanded={shareOpen}
                    aria-controls="arena-share-options"
                    onClick={() => setShareOpen((open) => !open)}
                  >
                    Share <span aria-hidden="true">{shareOpen ? "▾" : "▴"}</span>
                  </button>
                ) : (
                  <>
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
                  </>
                )}
                <button className="new-game-button" type="button" onClick={returnToSetup}>New game</button>
              </div>
            </section>
          )}
        </aside>
      </section>
    </main>
  );
}

function MoveCell({
  move,
  review,
  currentPly,
  onSelect,
}: {
  move?: MoveRecord;
  review?: MoveReview;
  currentPly: number;
  onSelect: (ply: number) => void;
}) {
  if (!move) return <span className="score-move empty" aria-hidden="true">—</span>;
  const reviewDetail = review?.judgement
    ? ` · ${judgementName(review.judgement)} · ${Math.round(review.winningChanceLoss)}% winning chance lost${review.bestMoveSan ? ` · best was ${review.bestMoveSan}` : ""}`
    : "";
  const detail = `${move.actor} · ${move.source}${move.elapsedMs === null ? "" : ` · ${Math.round(move.elapsedMs)} ms`}${reviewDetail}`;
  return (
    <button
      type="button"
      className={`score-move${review?.judgement ? ` ${review.judgement}` : ""}${move.ply === currentPly ? " current" : ""}`}
      aria-label={`${move.color === "w" ? "White" : "Black"} played ${move.san}. ${detail}`}
      aria-pressed={move.ply === currentPly}
      data-ply={move.ply}
      onClick={() => onSelect(move.ply)}
      title={detail}
    >
      <strong>{move.san}</strong>
      {review?.judgement ? (
        <>
          <ReviewPill judgement={review.judgement} />
          {review.winningChanceLoss >= 0.5 ? (
            <small className="review-loss">−{Math.round(review.winningChanceLoss)}%</small>
          ) : null}
        </>
      ) : null}
    </button>
  );
}

function GameReviewSummary({
  review,
  players,
  profiles,
  modelReferences,
  onRetry,
}: {
  review: ReviewState;
  players: Players;
  profiles: PlayerProfiles;
  modelReferences: PlayerModelReferences;
  onRetry: () => void;
}) {
  if (review.phase === "loading" || review.phase === "analyzing") {
    const completed = review.progress?.completed ?? 0;
    const total = review.progress?.total;
    return (
      <section className="review-progress" aria-live="polite">
        <div><i style={{ width: total ? `${(completed / total) * 100}%` : "8%" }} /></div>
        <span>{review.phase === "loading" ? "Loading Stockfish…" : `Analysing ${completed} of ${total} positions…`}</span>
      </section>
    );
  }
  if (review.phase === "error") {
    return (
      <section className="review-error" role="status">
        <span>{review.error ?? "The game could not be analysed."}</span>
        <button type="button" onClick={onRetry}>Retry</button>
      </section>
    );
  }
  if (!review.result) return null;
  return (
    <section
      className="review-summary"
      aria-label={`Post-game review by ${review.result.engine}`}
      title={`${review.result.engine} · ${review.result.nodesPerPosition.toLocaleString()} nodes per position`}
    >
      <PlayerReviewSummary color="White" name={players.w} profileId={profiles.w} modelReference={modelReferences.w} review={review.result.players.w} />
      <PlayerReviewSummary color="Black" name={players.b} profileId={profiles.b} modelReference={modelReferences.b} review={review.result.players.b} />
    </section>
  );
}

function PlayerReviewSummary({
  color,
  name,
  profileId,
  modelReference,
  review,
}: {
  color: string;
  name: string;
  profileId?: string | null;
  modelReference?: string | null;
  review: PlayerReview;
}) {
  return (
    <div className="review-player">
      <div className="review-player-heading">
        <span className="review-player-color">{color}</span>
        {modelReference
          ? <ModelNameWithCopy name={name} reference={modelReference} />
          : profileId ? <Link href={`/players/${profileId}`} title={name}>{name}</Link> : <b title={name}>{name}</b>}
      </div>
      <strong><b>{Math.round(review.accuracy)}%</b> accuracy</strong>
      <div className="review-categories">
        <ReviewGroup label="Good" judgements={POSITIVE_JUDGEMENTS} review={review} />
        <ReviewGroup label="Errors" judgements={NEGATIVE_JUDGEMENTS} review={review} />
      </div>
    </div>
  );
}

function ReviewGroup({
  label,
  judgements,
  review,
}: {
  label: string;
  judgements: readonly ReviewJudgement[];
  review: PlayerReview;
}) {
  return (
    <div className="review-group">
      <small>{label}</small>
      {judgements.map((judgement) => {
        const count = review.counts[judgement];
        const meta = JUDGEMENT_META[judgement];
        const countLabel = `${count} ${count === 1 ? meta.label.toLowerCase() : meta.plural}`;
        return (
          <div className={`review-stat${count > 0 ? "" : " inactive"}`} aria-label={countLabel} title={countLabel} key={judgement}>
            <ReviewPill judgement={judgement} inactive={count === 0} />
            <span>{meta.label}</span>
            <b>{count}</b>
          </div>
        );
      })}
    </div>
  );
}

function ReviewPill({ judgement, inactive = false }: { judgement: ReviewJudgement; inactive?: boolean }) {
  const meta = JUDGEMENT_META[judgement];
  return (
    <span
      className={`review-pill ${judgement} ${inactive ? "inactive" : "active"}`}
      aria-hidden="true"
    >
      {meta.glyph}
    </span>
  );
}

function judgementName(judgement: ReviewJudgement): string {
  return JUDGEMENT_META[judgement].label;
}

function ModelNameWithCopy({ name, reference }: { name: string; reference: string }) {
  const [copied, setCopied] = useState(false);
  const [revealed, setRevealed] = useState(false);

  const copyReference = async () => {
    try {
      await navigator.clipboard.writeText(reference);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1800);
    } catch {
      setCopied(false);
    }
  };

  return (
    <span className={`model-name-with-copy${copied ? " copied" : ""}${revealed ? " revealed" : ""}`}>
      <Link
        href={modelPageHref(reference)}
        title={name}
        onClick={(event) => {
          if (!revealed && window.matchMedia("(hover: none)").matches) {
            event.preventDefault();
            setRevealed(true);
          }
        }}
      >
        {name}
      </Link>
      <button
        className="model-name-copy"
        type="button"
        aria-label={`Copy full reference for ${name}`}
        title={copied ? "Reference copied" : "Copy full reference"}
        onClick={() => void copyReference()}
      >
        {copied ? (
          <span aria-hidden="true">✓</span>
        ) : (
          <svg aria-hidden="true" viewBox="0 0 16 16">
            <rect x="5" y="5" width="8" height="8" rx="1" />
            <path d="M3 11H2.5A1.5 1.5 0 0 1 1 9.5v-7A1.5 1.5 0 0 1 2.5 1h7A1.5 1.5 0 0 1 11 2.5V3" />
          </svg>
        )}
      </button>
    </span>
  );
}

function PgnExport({ pgn }: { pgn: string }) {
  const [copied, setCopied] = useState(false);

  const copyPgn = async () => {
    const done = () => {
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1800);
    };
    try {
      await navigator.clipboard.writeText(pgn);
      done();
    } catch {
      const holder = document.createElement("textarea");
      holder.value = pgn;
      holder.style.position = "fixed";
      holder.style.opacity = "0";
      document.body.append(holder);
      holder.select();
      const copiedViaCommand = document.execCommand("copy");
      holder.remove();
      if (copiedViaCommand) done();
      else setCopied(false);
    }
  };

  return (
    <section className="pgn-export" aria-label="Game PGN">
      <header>
        <span>PGN</span>
        <button type="button" onClick={() => void copyPgn()}>{copied ? "Copied ✓" : "Copy PGN"}</button>
      </header>
      <pre>{pgn}</pre>
    </section>
  );
}

function PlayerStrip({ color, name, profileId, modelReference, captured, lead }: PlayerStripProps) {
  const colorName = color === "w" ? "White" : "Black";
  const capturedColor = oppositeColor(color);

  return (
    <section className={`player-strip ${color === "w" ? "white" : "black"}`} aria-label={`${colorName} player`}>
      <div className="player-identity">
        <span className="player-color">{colorName}</span>
        {modelReference
          ? <ModelNameWithCopy name={name} reference={modelReference} />
          : profileId ? <Link href={`/players/${profileId}`}>{name}</Link> : <strong>{name}</strong>}
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
  moveTimeLimitMs,
  onMoveTimeLimitMs,
  onReference,
  onLoad,
}: {
  label: string;
  role: string;
  slot: ModelSlot;
  moveTimeLimitMs: number;
  onMoveTimeLimitMs: (value: number) => void;
  onReference: (reference: string) => void;
  onLoad: () => void;
}) {
  const inputId = `model-${label.toLowerCase().replace(" ", "-")}`;
  const capId = `${inputId}-thinking-cap`;
  const percent = slot.progress?.totalBytes
    ? Math.min(100, (slot.progress.loadedBytes / slot.progress.totalBytes) * 100)
    : null;
  return (
    <article className="model-loader">
      <header><span>{label}</span><small>{role}</small></header>
      <label htmlFor={inputId}>Hugging Face model</label>
      <div className="model-input-row">
        <input
          id={inputId}
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
      <label className="thinking-cap-field" htmlFor={capId}>
        <span>Thinking cap (ms)</span>
        <input
          id={capId}
          type="number"
          min={1}
          max={600000}
          required
          value={moveTimeLimitMs}
          onChange={(event) => onMoveTimeLimitMs(Number(event.target.value))}
        />
        <small>Per-move budget passed to the package.</small>
      </label>
      {slot.phase === "loading" && slot.progress ? (
        <div className="load-progress" aria-live="polite">
          <div><i style={{ width: percent === null ? "24%" : `${percent}%` }} /></div>
          <span>
            {slot.progress.label} · {formatBytes(slot.progress.loadedBytes)}
            {slot.progress.totalBytes ? ` / ${formatBytes(slot.progress.totalBytes)}` : ""}
          </span>
        </div>
      ) : null}
      {slot.model ? (
        <div className="model-summary">
          {slot.profileId
            ? <Link href={modelPageHref(slot.model.info.reference)}><strong>{slot.model.info.name}</strong></Link>
            : <strong>{slot.model.info.name}</strong>}
          <span>
            {slot.model.info.runtime} · {formatBytes(slot.model.info.artifactBytes)} ·{" "}
            {slot.model.info.pinned ? "Pinned" : "Mutable"}
          </span>
          <code title={`SHA-256 ${slot.model.info.digest}`}>
            SHA {slot.model.info.digest.slice(0, 12)}…
          </code>
        </div>
      ) : null}
      {slot.error ? <p className="model-error" role="alert">{slot.error}</p> : null}
    </article>
  );
}

async function resolveModelProfile(reference: string): Promise<{ id: string; name: string }> {
  const response = await fetch("/api/players/resolve", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ reference }),
  });
  const payload = await response.json() as { id?: unknown; name?: unknown; error?: unknown };
  if (!response.ok || typeof payload.id !== "string" || typeof payload.name !== "string") {
    throw new Error(typeof payload.error === "string" ? payload.error : "The model history profile could not be resolved.");
  }
  return { id: payload.id, name: payload.name };
}

function gameAtPly(moves: MoveRecord[], ply: number): Chess {
  const game = new Chess();
  for (const move of moves.slice(0, ply)) game.move(move.san);
  return game;
}

function describeHistoryPosition(ply: number, move?: MoveRecord): string {
  if (ply === 0 || !move) return "Reviewing starting position";
  const moveNumber = Math.ceil(ply / 2);
  return `Reviewing move ${moveNumber}${move.color === "w" ? "." : "…"} ${move.san}`;
}

function describeGame(game: Chess, running: boolean, thinking: string | null): string {
  if (game.isCheckmate()) return `${game.turn() === "w" ? "Black" : "White"} wins by checkmate`;
  if (game.isDraw()) return "Draw";
  if (game.isGameOver()) return "Game over";
  const side = game.turn() === "w" ? "White" : "Black";
  if (thinking) return `${side} to move · calculating`;
  return `${side} to move${game.isCheck() ? " · check" : ""}${running ? "" : " · paused"}`;
}

function pairMoves(moves: MoveRecord[]) {
  const rows: Array<{ number: number; white?: MoveRecord; black?: MoveRecord }> = [];
  for (const move of moves) {
    const index = Math.floor((move.ply - 1) / 2);
    const row = rows[index] ?? { number: index + 1 };
    if (move.color === "w") row.white = move;
    else row.black = move;
    rows[index] = row;
  }
  return rows;
}

function recordsFromGame(game: Chess, white: string, black: string): MoveRecord[] {
  return game.history({ verbose: true }).map((move, index) => ({
    ply: index + 1,
    san: move.san,
    actor: move.color === "w" ? white : black,
    source: "Shared PGN",
    elapsedMs: null,
    from: move.from,
    to: move.to,
    color: move.color,
    captured: move.captured,
  }));
}

function describePgnResult(game: Chess): string | null {
  if (game.isGameOver()) return null;
  const result = game.getHeaders().Result;
  if (result === "1-0") return "White wins";
  if (result === "0-1") return "Black wins";
  if (result === "1/2-1/2") return "Draw";
  return null;
}

function createSharePgn(
  game: Chess,
  players: Players,
  mode: PlayMode,
  player1Color: Color,
  humanColor: Color,
  gameError: string | null,
): string {
  const shared = new Chess();
  for (const san of game.history()) shared.move(san);
  const priorHeaders = game.getHeaders();
  const result = resultForShare(game, gameError);
  shared.setHeader("Event", "ChessGPT Arena");
  shared.setHeader("Site", window.location.origin);
  shared.setHeader("White", sanitizePgnHeader(players.w));
  shared.setHeader("Black", sanitizePgnHeader(players.b));
  shared.setHeader("Result", result);
  shared.setHeader("Mode", mode);
  shared.setHeader("PlayerOneColor", player1Color === "w" ? "White" : "Black");
  shared.setHeader("HumanColor", humanColor === "w" ? "White" : "Black");
  shared.setHeader(
    "Termination",
    sanitizePgnHeader(gameError ? `forfeit: ${gameError}` : priorHeaders.Termination ?? terminationForGame(game)),
  );
  return shared.pgn({ maxWidth: 0 });
}

function resultForShare(game: Chess, gameError: string | null): string {
  if (game.isCheckmate() || gameError) return game.turn() === "w" ? "0-1" : "1-0";
  if (game.isDraw()) return "1/2-1/2";
  return game.getHeaders().Result ?? "*";
}

function terminationForGame(game: Chess): string {
  if (game.isCheckmate()) return "checkmate";
  if (game.isStalemate()) return "stalemate";
  if (game.isThreefoldRepetition()) return "threefold repetition";
  if (game.isInsufficientMaterial()) return "insufficient material";
  if (game.isDrawByFiftyMoves()) return "fifty-move rule";
  return "draw";
}

function sanitizePgnHeader(value: string): string {
  return value.replace(/[\\\r\n]+/g, " ").replaceAll('"', "'").trim().slice(0, 160);
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
