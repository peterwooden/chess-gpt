"use client";

import { useEffect } from "react";

const POLL_INTERVAL_MS = 500;

export function TournamentPhaseWatcher({
  tournamentId,
  status,
}: {
  tournamentId: string;
  status: "registration" | "running" | "completed";
}) {
  useEffect(() => {
    let stopped = false;
    let refreshing = false;
    const refresh = async () => {
      if (refreshing || stopped) return;
      refreshing = true;
      try {
        const response = await fetch(`/api/tournaments/${tournamentId}/results`, {
          cache: "no-store",
        });
        if (!response.ok) return;
        const next = await response.json() as { status: typeof status };
        if (!stopped && next.status !== status) window.location.reload();
      } catch {
        // The next poll retries without disrupting the current phase UI.
      } finally {
        refreshing = false;
      }
    };
    const timer = window.setInterval(() => void refresh(), POLL_INTERVAL_MS);
    return () => {
      stopped = true;
      window.clearInterval(timer);
    };
  }, [status, tournamentId]);

  return null;
}
