export function selectModelVersion(versions, requestedCommit) {
  if (!requestedCommit) return versions[0] ?? null;
  return versions.find((version) => version.commitSha === requestedCommit) ?? null;
}
