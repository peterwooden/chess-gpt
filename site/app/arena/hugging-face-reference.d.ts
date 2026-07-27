export type ParsedHuggingFaceReference = { repository: string; revision: string };
export type ResolvedHuggingFaceReference = ParsedHuggingFaceReference & {
  reference: string;
  manifestUrl: string;
};

export function parseHuggingFaceReference(rawReference: string): ParsedHuggingFaceReference;
export function resolveHuggingFaceReference(
  rawReference: string,
  fetcher?: typeof fetch,
): Promise<ResolvedHuggingFaceReference>;
