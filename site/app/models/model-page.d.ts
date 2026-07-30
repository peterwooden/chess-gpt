export function selectModelVersion<T extends { commitSha: string }>(versions: readonly T[], requestedCommit?: string): T | null;
