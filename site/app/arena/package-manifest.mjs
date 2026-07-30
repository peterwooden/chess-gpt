export const PACKAGE_SCHEMA = "chess-gpt-package-v1";
export const PACKAGE_LIMIT_BYTES = 100_000_000;

export function parsePackageManifest(bytes) {
  const manifest = parseManifest(decodeJson(bytes));
  const declaredBytes = manifest.entrypoint.bytes
    + Object.values(manifest.artifacts).reduce((total, artifact) => total + artifact.bytes, 0);
  const packageBytes = bytes.byteLength + declaredBytes;
  if (packageBytes > PACKAGE_LIMIT_BYTES) {
    throw new Error("The package is over the 100 MB limit.");
  }
  return { manifest, packageBytes };
}

function parseManifest(value) {
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
  const artifacts = {};
  for (const [name, rawDescriptor] of Object.entries(value.artifacts)) {
    if (!name) throw new Error("Artifact names must be non-empty.");
    artifacts[name] = parseDescriptor(rawDescriptor, `artifact “${name}”`);
  }
  const paths = [entrypoint.path, ...Object.values(artifacts).map((item) => item.path)];
  if (new Set(paths).size !== paths.length) throw new Error("Package file paths must be unique.");
  return { schema: PACKAGE_SCHEMA, name: value.name, entrypoint, artifacts, config: value.config };
}

function parseDescriptor(value, label) {
  if (!isRecord(value)) throw new Error(`The ${label} descriptor is missing.`);
  if (!isSafeRelativePath(value.path)) {
    throw new Error(`The ${label} path must stay beneath browser/ and may not contain '..'.`);
  }
  if (!Number.isInteger(value.bytes) || value.bytes < 0) {
    throw new Error(`The ${label} must declare its exact byte length.`);
  }
  if (typeof value.sha256 !== "string" || !/^[0-9a-f]{64}$/.test(value.sha256)) {
    throw new Error(`The ${label} must declare a lowercase SHA-256 digest.`);
  }
  return { path: value.path, bytes: value.bytes, sha256: value.sha256 };
}

function isRecord(value) {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function isSafeRelativePath(value) {
  return typeof value === "string"
    && value.length > 0
    && !value.startsWith("/")
    && !value.includes("\\")
    && !value.includes("?")
    && !value.includes("#")
    && !value.split("/").some((part) => part === "" || part === "." || part === "..");
}

function decodeJson(bytes) {
  try {
    return JSON.parse(new TextDecoder().decode(bytes));
  } catch {
    throw new Error("The package manifest is not valid JSON.");
  }
}
