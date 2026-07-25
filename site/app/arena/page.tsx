import type { Metadata } from "next";
import ArenaClient from "./arena-client";

export const metadata: Metadata = {
  title: "ChessGPT Arena · Chess GPT Learning Lab",
  description: "Load revisioned chess models from Hugging Face and run human or model-versus-model games entirely in the browser.",
};

export default function ArenaPage() {
  return <ArenaClient />;
}
