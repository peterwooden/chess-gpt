import * as ort from "onnxruntime-web/webgpu";

type PackageGame = {
  chooseMove(input: { history: readonly string[]; legalMoves: readonly string[] }): Promise<string>;
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
};

let loadedPackage: LoadedPackage | null = null;
let currentGame: PackageGame | null = null;

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
  return currentGame.chooseMove({
    history: Object.freeze([...request.history]),
    legalMoves: Object.freeze([...request.legalMoves]),
  });
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
  const originalFetch = globalThis.fetch.bind(globalThis);
  globalThis.fetch = (input: RequestInfo | URL, init?: RequestInit) => {
    const rawUrl = input instanceof Request ? input.url : String(input);
    const url = new URL(rawUrl, self.location.href);
    if (url.origin !== self.location.origin) {
      return Promise.reject(new Error("Package network access is forbidden."));
    }
    return originalFetch(input, init);
  };
  for (const name of ["WebSocket", "EventSource", "XMLHttpRequest", "indexedDB", "caches"]) {
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
