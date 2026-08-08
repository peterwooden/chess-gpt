"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import type { TournamentStatus } from "../../../lib/tournaments";

const NEXT_STATUS: Record<TournamentStatus, { status: TournamentStatus; label: string; note: string }[]> = {
  registration: [{
    status: "running",
    label: "Close registration and start",
    note: "Entries freeze. Every entry must have passed its smoke test.",
  }],
  running: [
    { status: "completed", label: "Mark completed", note: "Publishes the final standings." },
    { status: "registration", label: "Reopen registration", note: "Recorded games are kept." },
  ],
  completed: [{
    status: "running",
    label: "Reopen for more games",
    note: "Resumes from the games already recorded.",
  }],
};

export function TournamentAdminControls({
  tournamentId,
  status,
}: {
  tournamentId: string;
  status: TournamentStatus;
}) {
  const router = useRouter();
  const [busy, setBusy] = useState<TournamentStatus | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function change(next: TournamentStatus) {
    setBusy(next);
    setError(null);
    try {
      const response = await fetch(`/api/tournaments/${tournamentId}/status`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ status: next }),
      });
      const payload = await response.json() as { error?: string };
      if (!response.ok) throw new Error(payload.error ?? "The status could not be changed.");
      router.refresh();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "The status could not be changed.");
    } finally {
      setBusy(null);
    }
  }

  return (
    <section className="tournament-panel">
      <h2>Tournament management</h2>
      <div className="tournament-admin-actions">
        {NEXT_STATUS[status].map((action) => (
          <button
            key={action.status}
            type="button"
            className={action.status === "running" ? "tournament-primary" : undefined}
            onClick={() => change(action.status)}
            disabled={busy !== null}
          >
            {busy === action.status ? "Working…" : action.label}
            <small>{action.note}</small>
          </button>
        ))}
      </div>
      {error ? <p className="tournament-error">{error}</p> : null}
    </section>
  );
}
