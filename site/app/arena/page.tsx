import type { Metadata } from "next";
import { getChatGPTUser } from "../chatgpt-auth";
import ArenaClient from "./arena-client";

export const dynamic = "force-dynamic";

export const metadata: Metadata = {
  title: "ChessGPT Arena · Chess GPT Learning Lab",
  description: "Load revisioned chess models from Hugging Face and run human or model-versus-model games entirely in the browser.",
};

export default async function ArenaPage() {
  const user = await getChatGPTUser();
  return <ArenaClient viewer={{ signedIn: Boolean(user), name: user?.fullName ?? null }} />;
}
