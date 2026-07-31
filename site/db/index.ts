import { drizzle } from "drizzle-orm/d1";
import * as schema from "./schema";

export type ChessGptRuntimeEnv = {
  DB?: D1Database;
  PLAYER_ID_HMAC_SECRET?: string;
  /** Comma-separated emails allowed to create tournaments and change their phase. */
  TOURNAMENT_ADMIN_EMAILS?: string;
};

export async function getRuntimeEnvironment(): Promise<ChessGptRuntimeEnv> {
  const { env } = await import("cloudflare:workers");
  return env as unknown as ChessGptRuntimeEnv;
}

export async function getDb() {
  const { DB } = await getRuntimeEnvironment();
  if (!DB) {
    throw new Error(
      "Cloudflare D1 binding `DB` is unavailable. Set the `d1` field in .openai/hosting.json to `DB` or let your control plane inject the real binding values before using the database."
    );
  }

  return drizzle(DB, { schema });
}

export async function getD1(): Promise<D1Database> {
  const { DB } = await getRuntimeEnvironment();
  if (!DB) throw new Error("Cloudflare D1 binding `DB` is unavailable.");
  return DB;
}
