export const PACKAGE_SCHEMA: "chess-gpt-package-v1";
export const PACKAGE_LIMIT_BYTES: 100000000;

export type ArtifactDescriptor = { path: string; sha256: string; bytes: number };
export type PackageManifest = {
  schema: typeof PACKAGE_SCHEMA;
  name: string;
  entrypoint: ArtifactDescriptor;
  artifacts: Record<string, ArtifactDescriptor>;
  config: unknown;
};

export function parsePackageManifest(bytes: Uint8Array): { manifest: PackageManifest; packageBytes: number };
