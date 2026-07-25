import * as ort from "onnxruntime-web/webgpu";

const MAX_ARTIFACT_BYTES = 150_000_000;
const HF_HOSTS = new Set(["huggingface.co", "www.huggingface.co"]);

export type LoadProgress = {
  stage: "manifest" | "vocabulary" | "model";
  loadedBytes: number;
  totalBytes: number | null;
};

export type ModelPrediction = {
  san: string;
  source: string;
};

export type BrowserModelInfo = {
  name: string;
  runtime: "SAN n-gram" | "ONNX · next SAN";
  sourceUrl: string;
  pinned: boolean;
  digest: string;
  artifactBytes: number;
};

export type BrowserChessModel = {
  info: BrowserModelInfo;
  predict(history: string[], legalMoves: string[]): Promise<ModelPrediction>;
  dispose(): Promise<void>;
};

type NormalizedReference = {
  artifactUrl: string;
  pinned: boolean;
};

type NgramState = {
  format_version: number;
  model_type: "san_backoff_ngram";
  order: number;
  metadata?: { experiment_id?: string };
  ngrams: Record<string, Record<string, Array<[string, number]>>>;
  side_counts: Record<string, Array<[string, number]>>;
};

type BrowserManifest = {
  schema: "chess-gpt-browser-v1";
  name: string;
  runtime: "onnx-next-san";
  context_length: number;
  model: ArtifactDescriptor;
  vocabulary: ArtifactDescriptor & {
    bos_token?: string;
    unknown_token?: string;
  };
  inputs: {
    input_ids: string;
    attention_mask?: string;
  };
  output: {
    logits: string;
  };
};

type ArtifactDescriptor = {
  path: string;
  sha256: string;
  bytes?: number;
};

type VocabularyFile = string[] | { tokens: string[] };

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
      artifactUrl: `https://huggingface.co/${owner}/${repository}/resolve/${revision}/browser/manifest.json`,
      pinned: isPinnedRevision(revision),
    };
  }

  const url = new URL(reference);
  if (!HF_HOSTS.has(url.hostname)) {
    throw new Error("For this first harness, model artifacts must come from huggingface.co.");
  }
  url.hostname = "huggingface.co";
  const parts = url.pathname.split("/").filter(Boolean);
  if (parts.length < 2) throw new Error("The Hugging Face URL must identify a model repository.");

  const [owner, repository, operation, revision] = parts;
  if (operation === "blob" && revision) {
    parts[2] = "resolve";
    url.pathname = `/${parts.join("/")}`;
    return { artifactUrl: url.toString(), pinned: isPinnedRevision(revision) };
  }
  if (operation === "resolve" && revision) {
    return { artifactUrl: url.toString(), pinned: isPinnedRevision(revision) };
  }
  if (operation === "tree" && revision) {
    return {
      artifactUrl: `https://huggingface.co/${owner}/${repository}/resolve/${revision}/browser/manifest.json`,
      pinned: isPinnedRevision(revision),
    };
  }
  if (parts.length === 2) {
    return {
      artifactUrl: `https://huggingface.co/${owner}/${repository}/resolve/main/browser/manifest.json`,
      pinned: false,
    };
  }
  throw new Error("Use a repository URL, a tree URL, or a direct resolve/blob artifact URL.");
}

export async function loadBrowserModel(
  rawReference: string,
  onProgress: (progress: LoadProgress) => void,
): Promise<BrowserChessModel> {
  const reference = normalizeModelReference(rawReference);
  const firstArtifact = await fetchBytes(reference.artifactUrl, "manifest", onProgress);

  if (new URL(reference.artifactUrl).pathname.endsWith(".gz")) {
    return loadNgramModel(firstArtifact, reference);
  }

  const decoded = decodeJson(firstArtifact, "model manifest");
  if (isNgramState(decoded)) {
    return createNgramModel(decoded, reference, firstArtifact.byteLength, await sha256Hex(firstArtifact));
  }
  if (!isBrowserManifest(decoded)) {
    throw new Error("Unsupported artifact. Expected a SAN n-gram checkpoint or chess-gpt-browser-v1 manifest.");
  }
  return loadOnnxModel(decoded, reference, onProgress);
}

async function loadNgramModel(
  compressed: Uint8Array,
  reference: NormalizedReference,
): Promise<BrowserChessModel> {
  if (typeof DecompressionStream === "undefined") {
    throw new Error("This browser cannot decompress the model. Use a current browser or an uncompressed JSON artifact.");
  }
  const stream = new Blob([compressed]).stream().pipeThrough(new DecompressionStream("gzip"));
  const decompressed = new Uint8Array(await new Response(stream).arrayBuffer());
  const decoded = decodeJson(decompressed, "SAN n-gram checkpoint");
  if (!isNgramState(decoded)) throw new Error("The checkpoint is not a supported SAN n-gram model.");
  return createNgramModel(decoded, reference, compressed.byteLength, await sha256Hex(compressed));
}

function createNgramModel(
  state: NgramState,
  reference: NormalizedReference,
  artifactBytes: number,
  digest: string,
): BrowserChessModel {
  return {
    info: {
      name: state.metadata?.experiment_id ?? "SAN backoff n-gram",
      runtime: "SAN n-gram",
      sourceUrl: reference.artifactUrl,
      pinned: reference.pinned,
      digest,
      artifactBytes,
    },
    async predict(history, legalMoves) {
      const legal = new Set(legalMoves);
      for (let order = Math.min(state.order, history.length); order > 0; order -= 1) {
        const context = history.slice(-order).join("\t");
        const best = bestLegal(state.ngrams[String(order)]?.[context] ?? [], legal);
        if (best) return { san: best, source: `${order}-move context` };
      }
      const side = String(history.length % 2);
      const best = bestLegal(state.side_counts[side] ?? [], legal);
      if (best) return { san: best, source: "side-to-move frequency" };
      return { san: [...legal].sort()[0], source: "deterministic legal fallback" };
    },
    async dispose() {},
  };
}

async function loadOnnxModel(
  manifest: BrowserManifest,
  reference: NormalizedReference,
  onProgress: (progress: LoadProgress) => void,
): Promise<BrowserChessModel> {
  const vocabularyUrl = siblingUrl(reference.artifactUrl, manifest.vocabulary.path);
  const vocabularyBytes = await fetchBytes(vocabularyUrl, "vocabulary", onProgress);
  await verifyArtifact(vocabularyBytes, manifest.vocabulary, "vocabulary");
  const vocabularyFile = decodeJson(vocabularyBytes, "SAN vocabulary") as VocabularyFile;
  const tokens = Array.isArray(vocabularyFile) ? vocabularyFile : vocabularyFile.tokens;
  if (!Array.isArray(tokens) || tokens.length === 0 || tokens.some((token) => typeof token !== "string")) {
    throw new Error("The vocabulary must be a non-empty JSON string array or { tokens: string[] }.");
  }

  const modelUrl = siblingUrl(reference.artifactUrl, manifest.model.path);
  const modelBytes = await fetchBytes(modelUrl, "model", onProgress);
  await verifyArtifact(modelBytes, manifest.model, "ONNX model");
  const tokenIds = new Map(tokens.map((token, index) => [token, index]));
  const useWebGpu = typeof navigator !== "undefined" && "gpu" in navigator;
  const session = await ort.InferenceSession.create(modelBytes, {
    executionProviders: useWebGpu ? ["webgpu", "wasm"] : ["wasm"],
  });

  for (const inputName of [manifest.inputs.input_ids, manifest.inputs.attention_mask].filter(Boolean)) {
    if (!session.inputNames.includes(inputName as string)) {
      await session.release();
      throw new Error(`The ONNX model does not expose the declared input “${inputName}”.`);
    }
  }
  if (!session.outputNames.includes(manifest.output.logits)) {
    await session.release();
    throw new Error(`The ONNX model does not expose the declared output “${manifest.output.logits}”.`);
  }

  const digest = await sha256Hex(modelBytes);
  return {
    info: {
      name: manifest.name,
      runtime: "ONNX · next SAN",
      sourceUrl: reference.artifactUrl,
      pinned: reference.pinned,
      digest,
      artifactBytes: modelBytes.byteLength + vocabularyBytes.byteLength,
    },
    async predict(history, legalMoves) {
      const unknownId = manifest.vocabulary.unknown_token
        ? tokenIds.get(manifest.vocabulary.unknown_token)
        : undefined;
      const historyIds = history.map((san) => tokenIds.get(san) ?? unknownId);
      if (historyIds.some((token) => token === undefined)) {
        const missing = history.find((san) => !tokenIds.has(san));
        throw new Error(`The model vocabulary cannot encode SAN move “${missing}”.`);
      }
      const bosId = manifest.vocabulary.bos_token
        ? tokenIds.get(manifest.vocabulary.bos_token)
        : undefined;
      if (manifest.vocabulary.bos_token && bosId === undefined) {
        throw new Error("The manifest's BOS token is absent from its vocabulary.");
      }
      const ids = [...(bosId === undefined ? [] : [bosId]), ...(historyIds as number[])].slice(
        -manifest.context_length,
      );
      if (ids.length === 0) {
        throw new Error("An empty history requires a vocabulary BOS token.");
      }

      const dimensions: [number, number] = [1, ids.length];
      const feeds: Record<string, ort.Tensor> = {
        [manifest.inputs.input_ids]: new ort.Tensor(
          "int64",
          BigInt64Array.from(ids, (value) => BigInt(value)),
          dimensions,
        ),
      };
      if (manifest.inputs.attention_mask) {
        feeds[manifest.inputs.attention_mask] = new ort.Tensor(
          "int64",
          BigInt64Array.from({ length: ids.length }, () => 1n),
          dimensions,
        );
      }
      const outputs = await session.run(feeds);
      const logits = outputs[manifest.output.logits];
      const vocabularySize = tokens.length;
      if (!logits || logits.dims.at(-1) !== vocabularySize) {
        throw new Error("The logits' final dimension does not match the SAN vocabulary.");
      }
      const offset = logits.data.length - vocabularySize;
      let chosen: string | null = null;
      let chosenLogit = Number.NEGATIVE_INFINITY;
      for (const san of legalMoves) {
        const tokenId = tokenIds.get(san);
        if (tokenId === undefined) continue;
        const logit = Number(logits.data[offset + tokenId]);
        if (logit > chosenLogit || (logit === chosenLogit && san > (chosen ?? ""))) {
          chosen = san;
          chosenLogit = logit;
        }
      }
      if (chosen) return { san: chosen, source: useWebGpu ? "ONNX · WebGPU" : "ONNX · WASM" };
      return { san: [...legalMoves].sort()[0], source: "deterministic legal fallback" };
    },
    async dispose() {
      await session.release();
    },
  };
}

async function fetchBytes(
  url: string,
  stage: LoadProgress["stage"],
  onProgress: (progress: LoadProgress) => void,
): Promise<Uint8Array> {
  const response = await fetch(url, { credentials: "omit", redirect: "follow" });
  if (!response.ok) throw new Error(`Hugging Face returned ${response.status} for ${stage}.`);
  const totalHeader = Number(response.headers.get("content-length"));
  const totalBytes = Number.isFinite(totalHeader) && totalHeader > 0 ? totalHeader : null;
  if (totalBytes !== null && totalBytes > MAX_ARTIFACT_BYTES) {
    throw new Error(`${stage} is larger than the 150 MB browser safety limit.`);
  }
  if (!response.body) {
    const bytes = new Uint8Array(await response.arrayBuffer());
    onProgress({ stage, loadedBytes: bytes.byteLength, totalBytes: bytes.byteLength });
    return bytes;
  }

  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let loadedBytes = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    loadedBytes += value.byteLength;
    if (loadedBytes > MAX_ARTIFACT_BYTES) {
      await reader.cancel();
      throw new Error(`${stage} exceeded the 150 MB browser safety limit.`);
    }
    chunks.push(value);
    onProgress({ stage, loadedBytes, totalBytes });
  }
  const bytes = new Uint8Array(loadedBytes);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return bytes;
}

function isPinnedRevision(revision: string): boolean {
  return /^[0-9a-f]{40}$/i.test(revision);
}

function siblingUrl(manifestUrl: string, relativePath: string): string {
  if (!relativePath || relativePath.startsWith("/") || relativePath.includes("..")) {
    throw new Error("Manifest artifact paths must be non-empty relative paths without '..'.");
  }
  const url = new URL(relativePath, manifestUrl);
  if (!HF_HOSTS.has(url.hostname)) throw new Error("Manifest artifacts must stay on huggingface.co.");
  return url.toString();
}

function decodeJson(bytes: Uint8Array, label: string): unknown {
  try {
    return JSON.parse(new TextDecoder().decode(bytes));
  } catch {
    throw new Error(`The ${label} is not valid JSON.`);
  }
}

function isNgramState(value: unknown): value is NgramState {
  if (!value || typeof value !== "object") return false;
  const state = value as Partial<NgramState>;
  return state.format_version === 1 && state.model_type === "san_backoff_ngram";
}

function isBrowserManifest(value: unknown): value is BrowserManifest {
  if (!value || typeof value !== "object") return false;
  const manifest = value as Partial<BrowserManifest>;
  return (
    manifest.schema === "chess-gpt-browser-v1" &&
    manifest.runtime === "onnx-next-san" &&
    typeof manifest.name === "string" &&
    Number.isInteger(manifest.context_length) &&
    (manifest.context_length ?? 0) > 0 &&
    Boolean(manifest.model && manifest.vocabulary && manifest.inputs && manifest.output)
  );
}

function bestLegal(pairs: Array<[string, number]>, legalMoves: Set<string>): string | null {
  let chosen: string | null = null;
  let chosenCount = -1;
  for (const [san, count] of pairs) {
    if (!legalMoves.has(san)) continue;
    if (count > chosenCount || (count === chosenCount && san > (chosen ?? ""))) {
      chosen = san;
      chosenCount = count;
    }
  }
  return chosen;
}

async function verifyArtifact(
  bytes: Uint8Array,
  descriptor: ArtifactDescriptor,
  label: string,
): Promise<void> {
  if (!/^[0-9a-f]{64}$/i.test(descriptor.sha256)) {
    throw new Error(`The ${label} must declare a full SHA-256 digest.`);
  }
  if (descriptor.bytes !== undefined && descriptor.bytes !== bytes.byteLength) {
    throw new Error(`The ${label} size does not match its manifest.`);
  }
  const actual = await sha256Hex(bytes);
  if (actual !== descriptor.sha256.toLowerCase()) {
    throw new Error(`The ${label} SHA-256 digest does not match its manifest.`);
  }
}

async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)].map((value) => value.toString(16).padStart(2, "0")).join("");
}
