import { init, parse, type ImportSpecifier } from "es-module-lexer";
import { createImmutableDownloadCache } from "./immutable-download-cache.mjs";
import { resolveHuggingFaceReference } from "./hugging-face-reference.mjs";
import {
  PACKAGE_LIMIT_BYTES,
  parsePackageManifest,
  type ArtifactDescriptor,
} from "./package-manifest.mjs";
import type { ThinkingCommand, ThinkingSample } from "../../lib/thinking-events.mjs";

const HF_HOSTS = new Set(["huggingface.co", "www.huggingface.co"]);
const immutableDownloadCache = createImmutableDownloadCache();

export type LoadProgress = {
  stage: "manifest" | "entrypoint" | "artifact";
  label: string;
  loadedBytes: number;
  totalBytes: number | null;
};

export type ModelPrediction = {
  san: string;
  source: string;
};

export type BrowserModelInfo = {
  name: string;
  runtime: "Package v1";
  sourceUrl: string;
  reference: string;
  repository: string;
  revision: string;
  pinned: boolean;
  digest: string;
  artifactBytes: number;
};

/**
 * Grace applied to the advertised per-move budget before the runner terminates
 * the worker. It exists so a package aiming at its budget is not forfeited for a
 * few milliseconds of overshoot; it is not additional thinking time.
 * See docs/TOURNAMENT_RULES.md.
 */
export const MOVE_TIME_GRACE_FACTOR = 1.25;

/** Per-move budget used for casual arena games when the caller omits a limit. */
export const DEFAULT_MOVE_TIME_LIMIT_MS = 10_000;

export function hardMoveLimitMs(moveTimeLimitMs: number): number {
  return Math.ceil(moveTimeLimitMs * MOVE_TIME_GRACE_FACTOR);
}

export type BrowserChessModel = {
  info: BrowserModelInfo;
  /**
   * False once the worker has been terminated, by a timeout or a crash. The
   * package forfeits only the game in progress, so a tournament runner must
   * reload a model that is no longer alive before its next game.
   */
  readonly alive: boolean;
  newGame(seed: number): Promise<void>;
  predict(
    history: string[],
    legalMoves: string[],
    moveTimeLimitMs?: number,
    onThinking?: (sample: ThinkingSample) => void,
  ): Promise<ModelPrediction>;
  dispose(): Promise<void>;
};

type WorkerResponse = {
  type?: "response";
  id: number;
  ok: boolean;
  value?: unknown;
  error?: string;
};

type WorkerThinking = {
  type: "thinking";
  id: number;
  elapsedMs: number;
  command: ThinkingCommand;
};

export async function loadBrowserModel(
  rawReference: string,
  onProgress: (progress: LoadProgress) => void,
): Promise<BrowserChessModel> {
  const reference = await resolveHuggingFaceReference(rawReference);
  const manifestBytes = await fetchBytes(
    reference.manifestUrl,
    "manifest",
    "manifest.json",
    PACKAGE_LIMIT_BYTES,
    onProgress,
  );
  const { manifest, packageBytes } = parsePackageManifest(manifestBytes);

  const entrypointUrl = packageFileUrl(reference.manifestUrl, manifest.entrypoint.path);
  const entrypointBytes = await immutableDownloadCache.load({
    url: entrypointUrl,
    immutable: true,
    maximumBytes: manifest.entrypoint.bytes,
    download: () => fetchBytes(
      entrypointUrl,
      "entrypoint",
      manifest.entrypoint.path,
      manifest.entrypoint.bytes,
      onProgress,
    ),
    validate: async (bytes) => {
      await verifyArtifact(bytes, manifest.entrypoint, "entrypoint");
      await assertSelfContainedEntrypoint(bytes);
    },
  });

  const artifacts: Array<{ name: string; bytes: Uint8Array }> = [];
  for (const [name, descriptor] of Object.entries(manifest.artifacts) as Array<[string, ArtifactDescriptor]>) {
    const artifactUrl = packageFileUrl(reference.manifestUrl, descriptor.path);
    const bytes = await immutableDownloadCache.load({
      url: artifactUrl,
      immutable: true,
      maximumBytes: descriptor.bytes,
      download: () => fetchBytes(
        artifactUrl,
        "artifact",
        name,
        descriptor.bytes,
        onProgress,
      ),
      validate: async (candidate) => {
        await verifyArtifact(candidate, descriptor, `artifact “${name}”`);
      },
    });
    artifacts.push({ name, bytes });
  }

  const worker = new Worker(new URL("./model-worker.ts", import.meta.url), { type: "module" });
  const client = createWorkerClient(worker);
  try {
    const artifactPayload = artifacts.map(({ name, bytes }) => ({ name, bytes: bytes.buffer }));
    await client.request(
      "load",
      {
        entrypoint: entrypointBytes.buffer,
        artifacts: artifactPayload,
        config: manifest.config,
      },
      [entrypointBytes.buffer, ...artifactPayload.map((artifact) => artifact.bytes)],
    );
  } catch (error) {
    client.terminate();
    throw error;
  }

  const digest = await sha256Hex(manifestBytes);
  return {
    info: {
      name: manifest.name,
      runtime: "Package v1",
      sourceUrl: reference.manifestUrl,
      reference: reference.reference,
      repository: reference.repository,
      revision: reference.revision,
      pinned: true,
      digest,
      artifactBytes: packageBytes,
    },
    get alive() {
      return client.alive;
    },
    async newGame(seed) {
      await client.request("newGame", { seed: seed >>> 0 });
    },
    async predict(history, legalMoves, moveTimeLimitMs = DEFAULT_MOVE_TIME_LIMIT_MS, onThinking) {
      const san = await client.request(
        "chooseMove",
        { history, legalMoves, moveTimeLimitMs },
        [],
        hardMoveLimitMs(moveTimeLimitMs),
        onThinking,
      );
      if (typeof san !== "string") throw new Error("The package returned a non-string move.");
      if (!legalMoves.includes(san)) throw new Error(`The package returned illegal SAN move “${san}”.`);
      return { san, source: "Package v1" };
    },
    async dispose() {
      try {
        await Promise.race([
          client.request("dispose", {}),
          new Promise((resolve) => window.setTimeout(resolve, 1_000)),
        ]);
      } finally {
        client.terminate();
      }
    },
  };
}

function createWorkerClient(worker: Worker) {
  let nextId = 1;
  let terminated = false;
  const pending = new Map<
    number,
    {
      resolve: (value: unknown) => void;
      reject: (error: Error) => void;
      onThinking?: (sample: ThinkingSample) => void;
    }
  >();

  function rejectPending(message: string) {
    for (const request of pending.values()) request.reject(new Error(message));
    pending.clear();
  }

  function failWorker(message: string) {
    if (terminated) return;
    terminated = true;
    worker.terminate();
    rejectPending(message);
  }

  worker.addEventListener("message", (event: MessageEvent<WorkerResponse | WorkerThinking>) => {
    const response = event.data;
    const request = pending.get(response.id);
    if (!request) return;
    if (response.type === "thinking") {
      try {
        request.onThinking?.({ elapsedMs: response.elapsedMs, command: response.command });
      } catch {
        // UI and broadcast listeners cannot fail a package move.
      }
      return;
    }
    pending.delete(response.id);
    if (response.ok) {
      request.resolve(response.value);
    } else {
      const message = response.error ?? "The package worker failed.";
      request.reject(new Error(message));
      failWorker(message);
    }
  });
  worker.addEventListener("error", (event) => {
    failWorker(event.message || "The package worker crashed.");
  });
  worker.addEventListener("messageerror", () => {
    failWorker("The package worker returned an unreadable message.");
  });

  return {
    get alive() {
      return !terminated;
    },
    request(
      type: string,
      payload: Record<string, unknown>,
      transfer: Transferable[] = [],
      timeoutMs?: number,
      onThinking?: (sample: ThinkingSample) => void,
    ) {
      if (terminated) return Promise.reject(new Error("The package worker has stopped."));
      const id = nextId;
      nextId += 1;
      return new Promise<unknown>((resolve, reject) => {
        // Terminating the worker is the only enforcement that survives a package
        // which blocks its worker synchronously: a busy worker never reads a
        // message, so cooperative cancellation cannot work here.
        const timer = timeoutMs === undefined
          ? undefined
          : setTimeout(() => {
            pending.delete(id);
            const message = `The package exceeded its ${timeoutMs} ms move time limit.`;
            reject(new Error(message));
            failWorker(message);
          }, timeoutMs);
        const settle = (finish: () => void) => {
          if (timer !== undefined) clearTimeout(timer);
          finish();
        };
        pending.set(id, {
          resolve: (value) => settle(() => resolve(value)),
          reject: (error) => settle(() => reject(error)),
          onThinking,
        });
        try {
          worker.postMessage({ id, type, ...payload }, transfer);
        } catch (error) {
          pending.delete(id);
          const message = error instanceof Error ? error.message : "The package request failed.";
          settle(() => reject(new Error(message)));
          failWorker(message);
        }
      });
    },
    terminate() {
      if (terminated) return;
      terminated = true;
      worker.terminate();
      rejectPending("The package worker has stopped.");
    },
  };
}

async function assertSelfContainedEntrypoint(bytes: Uint8Array): Promise<void> {
  await init;
  let imports: ReadonlyArray<ImportSpecifier>;
  try {
    [imports] = parse(new TextDecoder().decode(bytes));
  } catch {
    throw new Error("The package entrypoint is not valid JavaScript module syntax.");
  }
  if (imports.length > 0) {
    throw new Error("The package entrypoint must be self-contained and may not import other modules.");
  }
}

function packageFileUrl(manifestUrl: string, relativePath: string): string {
  const base = new URL(".", manifestUrl);
  const url = new URL(relativePath, base);
  if (!HF_HOSTS.has(url.hostname) || !url.pathname.startsWith(base.pathname)) {
    throw new Error("Package files must remain beneath browser/ in the same Hugging Face repository.");
  }
  return url.toString();
}

async function fetchBytes(
  url: string,
  stage: LoadProgress["stage"],
  label: string,
  maximumBytes: number,
  onProgress: (progress: LoadProgress) => void,
): Promise<Uint8Array> {
  const response = await fetch(url, {
    credentials: "omit",
    redirect: "follow",
    cache: "no-store",
  });
  if (!response.ok) throw new Error(`Hugging Face returned ${response.status} for ${label}.`);
  const totalHeader = Number(response.headers.get("content-length"));
  const totalBytes = Number.isFinite(totalHeader) && totalHeader > 0 ? totalHeader : null;
  if (totalBytes !== null && totalBytes > maximumBytes) {
    throw new Error(`${label} is larger than its allowed byte length.`);
  }
  if (!response.body) {
    const bytes = new Uint8Array(await response.arrayBuffer());
    if (bytes.byteLength > maximumBytes) throw new Error(`${label} exceeded its allowed byte length.`);
    onProgress({ stage, label, loadedBytes: bytes.byteLength, totalBytes: bytes.byteLength });
    return bytes;
  }

  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let loadedBytes = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    loadedBytes += value.byteLength;
    if (loadedBytes > maximumBytes) {
      await reader.cancel();
      throw new Error(`${label} exceeded its allowed byte length.`);
    }
    chunks.push(value);
    onProgress({ stage, label, loadedBytes, totalBytes: totalBytes ?? maximumBytes });
  }
  const bytes = new Uint8Array(loadedBytes);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return bytes;
}

async function verifyArtifact(
  bytes: Uint8Array,
  descriptor: ArtifactDescriptor,
  label: string,
): Promise<void> {
  if (descriptor.bytes !== bytes.byteLength) {
    throw new Error(`The ${label} size does not match its manifest.`);
  }
  const actual = await sha256Hex(bytes);
  if (actual !== descriptor.sha256) {
    throw new Error(`The ${label} SHA-256 digest does not match its manifest.`);
  }
}

async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const digestInput = new Uint8Array(bytes.byteLength);
  digestInput.set(bytes);
  const digest = await crypto.subtle.digest("SHA-256", digestInput);
  return [...new Uint8Array(digest)].map((value) => value.toString(16).padStart(2, "0")).join("");
}
