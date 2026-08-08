import * as ort from "onnxruntime-web/webgpu";
import runtimeWasmUrl from "onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm?url";
import {
  createThinkingCommandLimiter,
  normalizeThinkingCommand,
} from "../../lib/thinking-events.mjs";
import { provisionOnnxRuntime } from "./onnx-runtime-provisioning.mjs";

type PackageGame = {
  chooseMove(input: {
    history: readonly string[];
    legalMoves: readonly string[];
    moveTimeLimitMs: number;
    thinking: { emit(command: unknown): void };
  }): Promise<string>;
  dispose(): Promise<void>;
};

type LoadedPackage = {
  newGame(context: { random(): number }): Promise<PackageGame>;
  dispose(): Promise<void>;
};

type WorkerRequest = {
  id: number;
  type: "load" | "newGame" | "chooseMove" | "dispose";
  entrypoint?: ArrayBuffer;
  artifacts?: Array<{ name: string; bytes: ArrayBuffer }>;
  config?: unknown;
  seed?: number;
  history?: string[];
  legalMoves?: string[];
  moveTimeLimitMs?: number;
};

let loadedPackage: LoadedPackage | null = null;
let currentGame: PackageGame | null = null;
const monotonicNow = performance.now.bind(performance);

self.addEventListener("message", (event: MessageEvent<WorkerRequest>) => {
  void handleRequest(event.data);
});

async function handleRequest(request: WorkerRequest): Promise<void> {
  try {
    let value: unknown;
    if (request.type === "load") {
      value = await load(request);
    } else if (request.type === "newGame") {
      value = await newGame(request);
    } else if (request.type === "chooseMove") {
      value = await chooseMove(request);
    } else if (request.type === "dispose") {
      value = await dispose();
    } else {
      throw new Error("Unknown package-worker request.");
    }
    self.postMessage({ id: request.id, ok: true, value });
  } catch (error) {
    self.postMessage({
      id: request.id,
      ok: false,
      error: error instanceof Error ? error.message : "The package worker failed.",
    });
  }
}

async function load(request: WorkerRequest): Promise<void> {
  if (!(request.entrypoint instanceof ArrayBuffer) || !Array.isArray(request.artifacts)) {
    throw new Error("The package worker received incomplete files.");
  }
  await provisionOnnxRuntime({ ort, runtimeUrl: runtimeWasmUrl });
  restrictContestantCapabilities();
  const entrypointUrl = URL.createObjectURL(
    new Blob([request.entrypoint], { type: "text/javascript" }),
  );
  try {
    const adapterModule = await import(/* @vite-ignore */ entrypointUrl) as {
      loadPackage?: (context: {
        artifacts: ReadonlyMap<string, Uint8Array>;
        config: unknown;
        ort: typeof ort;
      }) => Promise<LoadedPackage>;
    };
    if (typeof adapterModule.loadPackage !== "function") {
      throw new Error("The entrypoint does not export loadPackage.");
    }
    const artifacts = new Map(
      request.artifacts.map(({ name, bytes }) => [name, new Uint8Array(bytes)]),
    );
    const candidate = await adapterModule.loadPackage({ artifacts, config: request.config, ort });
    if (!candidate || typeof candidate.newGame !== "function" || typeof candidate.dispose !== "function") {
      throw new Error("loadPackage returned an invalid package lifecycle.");
    }
    loadedPackage = candidate;
  } finally {
    URL.revokeObjectURL(entrypointUrl);
  }
}

async function newGame(request: WorkerRequest): Promise<void> {
  if (!loadedPackage) throw new Error("The package has not loaded.");
  if (currentGame) await currentGame.dispose();
  const candidate = await loadedPackage.newGame({ random: seededRandom(request.seed ?? 0) });
  if (!candidate || typeof candidate.chooseMove !== "function" || typeof candidate.dispose !== "function") {
    throw new Error("newGame returned an invalid game lifecycle.");
  }
  currentGame = candidate;
}

async function chooseMove(request: WorkerRequest): Promise<string> {
  if (!currentGame) throw new Error("The package game has not started.");
  if (!Array.isArray(request.history) || !Array.isArray(request.legalMoves)) {
    throw new Error("The runner did not supply SAN history and legal moves.");
  }
  if (!Number.isFinite(request.moveTimeLimitMs) || (request.moveTimeLimitMs ?? 0) <= 0) {
    throw new Error("The runner did not supply a per-move time limit.");
  }
  const startedAt = monotonicNow();
  const limiter = createThinkingCommandLimiter(monotonicNow);
  let active = true;
  const thinking = Object.freeze({
    emit(candidate: unknown) {
      if (!active) return;
      try {
        const command = normalizeThinkingCommand(candidate);
        if (!command || !limiter.accept()) return;
        self.postMessage({
          type: "thinking",
          id: request.id,
          elapsedMs: Math.max(0, monotonicNow() - startedAt),
          command,
        });
      } catch {
        // Cosmetic telemetry is deliberately non-throwing.
      }
    },
  });
  try {
    return await currentGame.chooseMove({
      history: Object.freeze([...request.history]),
      legalMoves: Object.freeze([...request.legalMoves]),
      moveTimeLimitMs: request.moveTimeLimitMs as number,
      thinking,
    });
  } finally {
    active = false;
  }
}

async function dispose(): Promise<void> {
  if (currentGame) await currentGame.dispose();
  currentGame = null;
  if (loadedPackage) await loadedPackage.dispose();
  loadedPackage = null;
}

function seededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4_294_967_296;
  };
}

function restrictContestantCapabilities(): void {
  globalThis.fetch = () => Promise.reject(new Error("Package network access is forbidden."));
  for (const name of [
    "WebSocket",
    "EventSource",
    "XMLHttpRequest",
    "importScripts",
    "indexedDB",
    "caches",
  ]) {
    try {
      Object.defineProperty(globalThis, name, {
        configurable: true,
        get() {
          throw new Error(`Package access to ${name} is forbidden.`);
        },
      });
    } catch {
      // The package is trusted; non-configurable browser globals remain governed by the rules.
    }
  }
}
