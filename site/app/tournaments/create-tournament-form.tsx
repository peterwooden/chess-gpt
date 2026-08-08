"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";

/**
 * Defaults chosen so a tournament is runnable without thinking, not because
 * they are the right contest. The move limit decides what the tournament is —
 * blitz or rapid — and the game count decides how confidently it is known.
 */
const DEFAULTS = {
  name: "",
  gamesPerPair: 100,
  moveTimeLimitMs: 300,
  residentBudgetMb: 1_500,
  maxAttemptsPerGame: 3,
};

export function CreateTournamentForm() {
  const router = useRouter();
  const [values, setValues] = useState(DEFAULTS);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState(false);

  function update<K extends keyof typeof DEFAULTS>(key: K, raw: string) {
    setValues((current) => ({
      ...current,
      [key]: key === "name" ? raw : Number(raw),
    }));
  }

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const response = await fetch("/api/tournaments", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: values.name,
          gamesPerPair: values.gamesPerPair,
          moveTimeLimitMs: values.moveTimeLimitMs,
          residentBudgetBytes: Math.round(values.residentBudgetMb * 1_000_000),
          maxAttemptsPerGame: values.maxAttemptsPerGame,
        }),
      });
      const payload = await response.json() as { tournament?: { id: string }; error?: string };
      if (!response.ok || !payload.tournament) {
        throw new Error(payload.error ?? "The tournament could not be created.");
      }
      router.push(`/tournaments/${payload.tournament.id}`);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "The tournament could not be created.");
    } finally {
      setBusy(false);
    }
  }

  if (!open) {
    return (
      <section className="tournament-panel">
        <button type="button" className="tournament-primary" onClick={() => setOpen(true)}>
          Add tournament
        </button>
      </section>
    );
  }

  return (
    <section className="tournament-panel">
      <form className="tournament-form" onSubmit={submit}>
        <h2>New tournament</h2>
        <label>
          <span>Name</span>
          <input
            required
            value={values.name}
            onChange={(event) => update("name", event.target.value)}
            placeholder="Championship 2026"
          />
        </label>
        <label>
          <span>Games per pair</span>
          <input
            type="number" min={1} max={5000} required
            value={values.gamesPerPair}
            onChange={(event) => update("gamesPerPair", event.target.value)}
          />
          <small>How confidently the result is known.</small>
        </label>
        <label>
          <span>Move time limit (ms)</span>
          <input
            type="number" min={1} max={600000} required
            value={values.moveTimeLimitMs}
            onChange={(event) => update("moveTimeLimitMs", event.target.value)}
          />
          <small>What the contest is. Packages are told this budget.</small>
        </label>
        <label>
          <span>Resident package budget (MB)</span>
          <input
            type="number" min={1} max={8000} required
            value={values.residentBudgetMb}
            onChange={(event) => update("residentBudgetMb", event.target.value)}
          />
          <small>Above this, games run pair-major instead of interleaved.</small>
        </label>
        <label>
          <span>Attempts per game</span>
          <input
            type="number" min={1} max={20} required
            value={values.maxAttemptsPerGame}
            onChange={(event) => update("maxAttemptsPerGame", event.target.value)}
          />
          <small>Then the game is abandoned rather than retried forever.</small>
        </label>

        {error ? <p className="tournament-error">{error}</p> : null}

        <div className="tournament-form-actions">
          <button type="submit" className="tournament-primary" disabled={busy}>
            {busy ? "Creating…" : "Create tournament"}
          </button>
          <button type="button" onClick={() => setOpen(false)} disabled={busy}>Cancel</button>
        </div>
      </form>
    </section>
  );
}
