"use client";

import { useState } from "react";
import { modelChallengeHref } from "../arena/share-url.mjs";

export function ReferenceActions({ reference, compact = false }: { reference: string; compact?: boolean }) {
  const [message, setMessage] = useState("");

  async function copyReference() {
    try {
      await navigator.clipboard.writeText(reference);
      setMessage("Copied");
    } catch {
      setMessage("Copy failed");
    }
  }

  return (
    <div className={`reference-actions${compact ? " compact" : ""}`}>
      <button type="button" onClick={() => void copyReference()}>Copy reference</button>
      <a href={modelChallengeHref(reference)}>Challenge</a>
      <span aria-live="polite">{message}</span>
    </div>
  );
}
