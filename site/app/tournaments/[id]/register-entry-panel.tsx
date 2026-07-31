"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { loadBrowserModel } from "../../arena/model";
import { playGame } from "../../arena/play-game.mjs";
import type { TournamentStatus } from "../../../lib/tournaments";

export type EntrySummary = {
  id: string;
  displayName: string;
  reference: string;
  ownerPlayerId: string;
  verifiedAt: number | null;
  smokeMedianMs: number | null;
  packageBytes: number;
};

/** Plies played against the package's own moves to prove it actually runs. */
const SMOKE_PLIES = 20;
const SMOKE_MOVE_TIME_LIMIT_MS = 30_000;

type Phase = "idle" | "loading" | "smoke" | "saving";

export function RegisterEntryPanel({
  tournamentId,
  status,
  entries,
  viewerPlayerId,
  administrator,
  signedIn,
}: {
  tournamentId: string;
  status: TournamentStatus;
  entries: EntrySummary[];
  viewerPlayerId: string | null;
  administrator: boolean;
  signedIn: boolean;
}) {
  const router = useRouter();
  const [reference, setReference] = useState("");
  const [phase, setPhase] = useState<Phase>("idle");
  const [progress, setProgress] = useState<string>("");
  const [error, setError] = useState<string | null>(null);

  const open = status === "registration";

  /**
   * Registration is not a form submit. The package is downloaded, hash-verified
   * and actually played here, in the registrant's own browser, so a broken
   * submission is found days early rather than on the morning of the tournament.
   */
  async function register(event: React.FormEvent) {
    event.preventDefault();
    setError(null);
    setPhase("loading");
    setProgress("Resolving reference…");

    let model: Awaited<ReturnType<typeof loadBrowserModel>> | null = null;
    try {
      model = await loadBrowserModel(reference.trim(), (update) => {
        const percent = update.totalBytes
          ? ` ${Math.round((update.loadedBytes / update.totalBytes) * 100)}%`
          : "";
        setProgress(`Downloading ${update.label}${percent}`);
      });

      setPhase("smoke");
      setProgress("Playing a smoke game…");
      const self = {
        name: model.info.name,
        newGame: (seed: number) => model!.newGame(seed),
        predict: (history: string[], legalMoves: string[], limit: number) =>
          model!.predict(history, legalMoves, limit),
      };
      // The package plays both sides: this proves it loads, returns legal SAN
      // and keeps working across moves, which is all registration needs to know.
      const outcome = await playGame(self, self, {
        moveTimeLimitMs: SMOKE_MOVE_TIME_LIMIT_MS,
        maxPlies: SMOKE_PLIES,
        seed: 1,
      });
      if (outcome.failure) {
        throw new Error(`The package failed its smoke test: ${outcome.failure.reason}`);
      }
      if (outcome.moves.length === 0) {
        throw new Error("The package did not produce any moves.");
      }

      const timings = outcome.moves.map((move) => move.elapsedMs).sort((a, b) => a - b);
      const median = timings[Math.floor(timings.length / 2)];
      const p95 = timings[Math.min(timings.length - 1, Math.floor(timings.length * 0.95))];

      setPhase("saving");
      setProgress("Registering…");
      const response = await fetch(`/api/tournaments/${tournamentId}/entries`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          reference: model.info.reference,
          packageBytes: model.info.artifactBytes,
          smokeMoveCount: outcome.moves.length,
          smokeMedianMs: median,
          smokeP95Ms: p95,
        }),
      });
      const payload = await response.json() as { error?: string };
      if (!response.ok) throw new Error(payload.error ?? "The entry could not be registered.");

      setReference("");
      router.refresh();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "The entry could not be registered.");
    } finally {
      await model?.dispose();
      setPhase("idle");
      setProgress("");
    }
  }

  async function withdraw(entryId: string) {
    setError(null);
    try {
      const response = await fetch(`/api/tournaments/${tournamentId}/entries/${entryId}`, {
        method: "DELETE",
      });
      if (!response.ok) {
        const payload = await response.json() as { error?: string };
        throw new Error(payload.error ?? "The entry could not be withdrawn.");
      }
      router.refresh();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "The entry could not be withdrawn.");
    }
  }

  return (
    <section className="tournament-panel" aria-labelledby="entries-title">
      <h2 id="entries-title">Entries</h2>

      {entries.length === 0 ? (
        <p className="tournament-note">Nothing registered yet.</p>
      ) : (
        <ul className="tournament-entries">
          {entries.map((entry) => (
            <li key={entry.id}>
              <div>
                <strong>{entry.displayName}</strong>
                <code>{entry.reference}</code>
                <small>
                  {entry.verifiedAt ? "Verified" : "Not verified"}
                  {entry.smokeMedianMs !== null ? ` · ${entry.smokeMedianMs} ms median move` : ""}
                  {" · "}{(entry.packageBytes / 1_000_000).toFixed(1)} MB
                </small>
              </div>
              {open && (entry.ownerPlayerId === viewerPlayerId || administrator) ? (
                <button type="button" onClick={() => withdraw(entry.id)}>Withdraw</button>
              ) : null}
            </li>
          ))}
        </ul>
      )}

      {!open ? (
        <p className="tournament-note">
          Entries are frozen. They were fixed when the tournament left registration.
        </p>
      ) : !signedIn ? (
        <p className="tournament-note">Sign in to register a model.</p>
      ) : (
        <form className="tournament-register" onSubmit={register}>
          <label>
            <span>Hugging Face reference</span>
            <input
              required
              value={reference}
              onChange={(event) => setReference(event.target.value)}
              placeholder="owner/repository@revision"
              disabled={phase !== "idle"}
            />
            <small>
              Downloaded, hash-verified and played here before it is accepted. The
              reference is pinned to a commit, so publishing new weights later cannot
              change what runs.
            </small>
          </label>
          <button type="submit" className="tournament-primary" disabled={phase !== "idle"}>
            {phase === "idle" ? "Verify and register" : progress || "Working…"}
          </button>
        </form>
      )}

      {error ? <p className="tournament-error">{error}</p> : null}
    </section>
  );
}
