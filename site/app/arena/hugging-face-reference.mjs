const HF_HOSTS = new Set(["huggingface.co", "www.huggingface.co"]);
const COMMIT_SHA = /^[0-9a-f]{40}$/;

export function parseHuggingFaceReference(rawReference) {
  const reference = rawReference.trim();
  if (!reference) throw new Error("Enter a Hugging Face model URL or owner/repository@revision.");

  if (!reference.startsWith("http://") && !reference.startsWith("https://")) {
    const match = /^([^/@\s]+)\/([^@/\s]+)(?:@(.+))?$/.exec(reference);
    if (!match) throw new Error("Use owner/repository@revision or a huggingface.co model URL.");
    return { repository: `${match[1]}/${match[2]}`, revision: match[3]?.trim() || "main" };
  }

  const url = new URL(reference);
  if (!HF_HOSTS.has(url.hostname)) {
    throw new Error("Model packages must come from a public huggingface.co repository.");
  }
  const parts = url.pathname.split("/").filter(Boolean);
  if (parts.length < 2) throw new Error("The Hugging Face URL must identify a model repository.");
  const repository = `${parts[0]}/${parts[1]}`;
  const operation = parts[2];
  if (!operation) return { repository, revision: "main" };
  if ((operation === "tree" || operation === "blob" || operation === "resolve") && parts[3]) {
    return { repository, revision: decodeURIComponent(parts[3]) };
  }
  throw new Error("Use a repository URL, tree URL, or direct browser/manifest.json URL.");
}

export async function resolveHuggingFaceReference(rawReference, fetcher = fetch) {
  const parsed = parseHuggingFaceReference(rawReference);
  const endpoint = `https://huggingface.co/api/models/${parsed.repository}/revision/${encodeURIComponent(parsed.revision)}?expand%5B%5D=sha`;
  const response = await fetcher(endpoint, { headers: { Accept: "application/json" } });
  if (!response.ok) {
    if (response.status === 404) throw new Error("That Hugging Face model or revision does not exist.");
    throw new Error(`Hugging Face could not resolve this revision (HTTP ${response.status}).`);
  }
  const payload = await response.json();
  if (typeof payload?.id !== "string" || !COMMIT_SHA.test(payload?.sha)) {
    throw new Error("Hugging Face returned an invalid model revision.");
  }
  const repository = payload.id;
  const revision = payload.sha;
  return {
    repository,
    revision,
    reference: `${repository}@${revision}`,
    manifestUrl: `https://huggingface.co/${repository}/resolve/${revision}/browser/manifest.json`,
  };
}
