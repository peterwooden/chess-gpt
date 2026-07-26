const PACKAGE_SCHEMA = "chess-gpt-package-v1";
const PACKAGE_LIMIT_BYTES = 100_000_000;
const HF_HOSTS = new Set(["huggingface.co", "www.huggingface.co"]);

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
  pinned: boolean;
  digest: string;
  artifactBytes: number;
};

export type BrowserChessModel = {
  info: BrowserModelInfo;
  newGame(seed: number): Promise<void>;
  predict(history: string[], legalMoves: string[]): Promise<ModelPrediction>;
  dispose(): Promise<void>;
};

type NormalizedReference = {
  manifestUrl: string;
  pinned: boolean;
};

type ArtifactDescriptor = {
  path: string;
  sha256: string;
  bytes: number;
};

type PackageManifest = {
  schema: typeof PACKAGE_SCHEMA;
  name: string;
  entrypoint: ArtifactDescriptor;
  artifacts: Record<string, ArtifactDescriptor>;
  config: unknown;
};

type WorkerResponse = {
  id: number;
  ok: boolean;
  value?: unknown;
  error?: string;
};

export function normalizeModelReference(rawReference: string): NormalizedReference {
  const reference = rawReference.trim();
  if (!reference) throw new Error("Enter a Hugging Face model URL or owner/repository@revision.");

  if (!reference.startsWith("http://") && !reference.startsWith("https://")) {
    const match = /^([^/@\s]+)\/([^@/\s]+)(?:@([^\s]+))?$/.exec(reference);
    if (!match) {
      throw new Error("Use owner/repository@revision or a huggingface.co model URL.");
    }
    const [, owner, repository, revision = "main"] = match;
    return {
      manifestUrl: `https://huggingface.co/${owner}/${repository}/resolve/${revision}/browser/manifest.json`,
      pinned: isPinnedRevision(revision),
    };
  }

  const url = new URL(reference);
  if (!HF_HOSTS.has(url.hostname)) {
    throw new Error("Model packages must come from a public huggingface.co repository.");
  }
  url.hostname = "huggingface.co";
  const parts = url.pathname.split("/").filter(Boolean);
  if (parts.length < 2) throw new Error("The Hugging Face URL must identify a model repository.");

  const [owner, repository, operation, revision] = parts;
  if ((operation === "blob" || operation === "resolve") && revision) {
    parts[2] = "resolve";
    url.pathname = `/${parts.join("/")}`;
    return { manifestUrl: url.toString(), pinned: isPinnedRevision(revision) };
  }
  if (operation === "tree" && revision) {
    return {
      manifestUrl: `https://huggingface.co/${owner}/${repository}/resolve/${revision}/browser/manifest.json`,
      pinned: isPinnedRevision(revision),
    };
  }
  if (parts.length === 2) {
    return {
      manifestUrl: `https://huggingface.co/${owner}/${repository}/resolve/main/browser/manifest.json`,
      pinned: false,
    };
  }
  throw new Error("Use a repository URL, tree URL, or direct browser/manifest.json URL.");
}

export async function loadBrowserModel(
  rawReference: string,
  onProgress: (progress: LoadProgress) => void,
): Promise<BrowserChessModel> {
  const reference = normalizeModelReference(rawReference);
  const manifestBytes = await fetchBytes(
    reference.manifestUrl,
    "manifest",
    "manifest.json",
    PACKAGE_LIMIT_BYTES,
    onProgress,
  );
  const decoded = decodeJson(manifestBytes, "package manifest");
  const manifest = parseManifest(decoded);
  const declaredBytes = manifest.entrypoint.bytes
    + Object.values(manifest.artifacts).reduce((total, artifact) => total + artifact.bytes, 0);
  const packageBytes = manifestBytes.byteLength + declaredBytes;
  if (packageBytes > PACKAGE_LIMIT_BYTES) {
    throw new Error(`The package is ${formatBytes(packageBytes)}, over the 100 MB limit.`);
  }

  const entrypointUrl = packageFileUrl(reference.manifestUrl, manifest.entrypoint.path);
  const entrypointBytes = await fetchBytes(
    entrypointUrl,
    "entrypoint",
    manifest.entrypoint.path,
    manifest.entrypoint.bytes,
    onProgress,
  );
  await verifyArtifact(entrypointBytes, manifest.entrypoint, "entrypoint");

  const artifacts: Array<{ name: string; bytes: Uint8Array }> = [];
  for (const [name, descriptor] of Object.entries(manifest.artifacts)) {
    const bytes = await fetchBytes(
      packageFileUrl(reference.manifestUrl, descriptor.path),
      "artifact",
      name,
      descriptor.bytes,
      onProgress,
    );
    await verifyArtifact(bytes, descriptor, `artifact “${name}”`);
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
      pinned: reference.pinned,
      digest,
      artifactBytes: packageBytes,
    },
    async newGame(seed) {
      await client.request("newGame", { seed: seed >>> 0 });
    },
    async predict(history, legalMoves) {
      const san = await client.request("chooseMove", { history, legalMoves });
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
    { resolve: (value: unknown) => void; reject: (error: Error) => void }
  >();

  function rejectPending(message: string) {
    for (const request of pending.values()) request.reject(new Error(message));
    pending.clear();
  }

  worker.addEventListener("message", (event: MessageEvent<WorkerResponse>) => {
    const response = event.data;
    const request = pending.get(response.id);
    if (!request) return;
    pending.delete(response.id);
    if (response.ok) request.resolve(response.value);
    else request.reject(new Error(response.error ?? "The package worker failed."));
  });
  worker.addEventListener("error", (event) => {
    rejectPending(event.message || "The package worker crashed.");
  });
  worker.addEventListener("messageerror", () => {
    rejectPending("The package worker returned an unreadable message.");
  });

  return {
    request(type: string, payload: Record<string, unknown>, transfer: Transferable[] = []) {
      if (terminated) return Promise.reject(new Error("The package worker has stopped."));
      const id = nextId;
      nextId += 1;
      return new Promise<unknown>((resolve, reject) => {
        pending.set(id, { resolve, reject });
        worker.postMessage({ id, type, ...payload }, transfer);
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

function parseManifest(value: unknown): PackageManifest {
  if (!isRecord(value) || value.schema !== PACKAGE_SCHEMA) {
    throw new Error(`Unsupported package. Expected schema ${PACKAGE_SCHEMA}.`);
  }
  if (typeof value.name !== "string" || !value.name.trim()) {
    throw new Error("The package manifest requires a non-empty name.");
  }
  const entrypoint = parseDescriptor(value.entrypoint, "entrypoint");
  if (!isRecord(value.artifacts) || !("config" in value)) {
    throw new Error("The package manifest requires artifacts and config fields.");
  }

  const artifacts: Record<string, ArtifactDescriptor> = {};
  for (const [name, rawDescriptor] of Object.entries(value.artifacts)) {
    if (!name) throw new Error("Artifact names must be non-empty.");
    artifacts[name] = parseDescriptor(rawDescriptor, `artifact “${name}”`);
  }
  const paths = [entrypoint.path, ...Object.values(artifacts).map((item) => item.path)];
  if (new Set(paths).size !== paths.length) throw new Error("Package file paths must be unique.");

  return {
    schema: PACKAGE_SCHEMA,
    name: value.name,
    entrypoint,
    artifacts,
    config: value.config,
  };
}

function parseDescriptor(value: unknown, label: string): ArtifactDescriptor {
  if (!isRecord(value)) throw new Error(`The ${label} descriptor is missing.`);
  if (!isSafeRelativePath(value.path)) {
    throw new Error(`The ${label} path must stay beneath browser/ and may not contain '..'.`);
  }
  if (!Number.isInteger(value.bytes) || (value.bytes as number) < 0) {
    throw new Error(`The ${label} must declare its exact byte length.`);
  }
  if (typeof value.sha256 !== "string" || !/^[0-9a-f]{64}$/.test(value.sha256)) {
    throw new Error(`The ${label} must declare a lowercase SHA-256 digest.`);
  }
  return { path: value.path as string, bytes: value.bytes as number, sha256: value.sha256 };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function isSafeRelativePath(value: unknown): value is string {
  return typeof value === "string"
    && value.length > 0
    && !value.startsWith("/")
    && !value.includes("\\")
    && !value.includes("?")
    && !value.includes("#")
    && !value.split("/").some((part) => part === "" || part === "." || part === "..");
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
  const response = await fetch(url, { credentials: "omit", redirect: "follow" });
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
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)].map((value) => value.toString(16).padStart(2, "0")).join("");
}

function decodeJson(bytes: Uint8Array, label: string): unknown {
  try {
    return JSON.parse(new TextDecoder().decode(bytes));
  } catch {
    throw new Error(`The ${label} is not valid JSON.`);
  }
}

function isPinnedRevision(revision: string): boolean {
  return /^[0-9a-f]{40}$/i.test(revision);
}

function formatBytes(bytes: number): string {
  return `${(bytes / 1_000_000).toFixed(1)} MB`;
}
