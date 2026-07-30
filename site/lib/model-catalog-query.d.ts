export type ModelCatalogCursor = { value: string; id: string };
export function buildModelDirectoryQuery(options: {
  search?: string;
  sort?: "recent" | "name" | "games" | "versions";
  cursor?: ModelCatalogCursor | null;
  limit?: number;
}): { sql: string; bindings: unknown[] };
export function buildModelProfileQuery(): string;
export const MODEL_VERSIONS_SQL: string;
