# Tournament harness design

Agreed 2026-07-31, ahead of tournament day. This document records how the site runs a
tournament and why each choice was made. The normative contract lives in
[`TOURNAMENT_RULES.md`](TOURNAMENT_RULES.md); this file is the engineering design that
implements it.

## What already exists

The browser arena can already do most of the hard parts. It resolves a Hugging Face
reference to an immutable commit SHA, downloads and SHA-256 verifies every declared file,
rejects an entrypoint that imports anything, runs the package inside a dedicated Worker with
network and storage capabilities revoked, and records finished games to D1 through
`lib/history.ts`.

What it cannot do is run a tournament. It plays one game at a time with a human watching,
has no schedule, no standings, no resume, and no ply cap.

## Design decisions

### Entries and nomination

A competitor may register any number of model versions. The title is decided by a
tournament that each competitor enters with exactly one nominated version.

The alternative — every model an entrant, highest score wins — was rejected because it
rewards submitting many variants. Scores carry a standard error of roughly five points over
a hundred games, so a competitor fielding five mediocre models has a good chance that one
posts a lucky result a single stronger model would not. That is the tournament-level version
of the test-set overfitting this repository exists to avoid.

Entries are per-tournament rather than global with a champion flag, so nomination is a
concrete act rather than a piece of metadata. Two tournaments are expected on the day: a
championship over the three nominated models, then an open tournament over everything
registered. Only tournament configuration distinguishes them.

### Sampled openings

Agreed 2026-08-08 as a per-tournament option beside the standard-start protocol of
2026-07-31: each tournament chooses at creation whether to use the opening book, and the
choice is displayed with its configuration. The standard-start protocol
accepted a known cost: both published candidates select moves deterministically (`argmax`,
never calling the supplied `random()`), so a pairing produced exactly two distinct games —
one per color — and every further game was a replay. Sample size within a pairing depended
entirely on a competitor choosing to make their submission stochastic, and none did.

Opening sampling restores sample size without touching the package interface. When an
opening-book tournament leaves registration, `sampleOpenings` draws `gamesPerPair / 2` openings from the
fixed book in `site/lib/opening-book.mjs` — established theory lines truncated at varying
even ply depths — and the draw is stored as JSON on the tournament row. Storing it, rather
than re-deriving it, keeps the runner plan a pure function of the tournament row and means a
later edit to the book cannot silently change a tournament already underway.

Book lines do not need to be objectively equal, only famous: because both models play each
side of every sampled opening, an uneven line is symmetric, and deliberately sharp or
dubious gambits (Englund, Halloween, Latvian, Grob) are kept in as a test in themselves —
a stronger model should hold the worse side of an uneven position where a weaker model
cannot, and the color-reversed pair converts exactly that difference into points.

The draw happens at the registration→running transition, not at creation, so entries are
frozen before anyone can see which openings will be played. Colors already alternate on the
game index, so opening `⌊gameIndex / 2⌋` gives each opening one game per color with no
schedule change; games per pair must be even for this to come out balanced (enforced only
when the book is on).

The runner plays the book moves itself before either package moves. They are recorded with
actor `book` and zero elapsed time, appear in the PGN with an `Opening` header, and reach
packages as ordinary `history`, so packages published before this change remain valid. An
illegal book move throws — it is a harness bug, never a competitor's forfeit.

Tournaments with the book off — and all tournaments created before the column existed —
have `openings = NULL` and play from the standard start.

The old tiebreak rule (more opening pairs on a tie) stays retired: ties still share the
title. The `distinctGamesByPair` metric now mostly reflects opening variety rather than
package stochasticity, since different openings trivially produce different games.

### Per-move time limit

Each tournament configures a per-move wall-clock limit, which is passed to the package as
`moveTimeLimitMs` so it can budget its own search. The runner enforces it independently by
terminating the worker at 1.25 times the limit, which is the only mechanism that survives a
package that blocks its worker synchronously.

A per-move inference-FLOP budget was considered and rejected for simplicity. It was
attractive — the runner owns the `ort` object it passes to `loadPackage`, so proxying
`InferenceSession.create` and `session.run` would measure ONNX work exactly, and combined
with static graph analysis at registration it would have reused the already-ratified
training profiler. It would also have made results hardware-independent and reproducible.
It was not adopted because it does not bound wall clock, it cannot see compute done in plain
JavaScript, and a clock is easier to reason about on the day.

Because the clock decides outcomes, hardware now matters, sequential execution is required,
and the runner machine must stay idle for the duration.

### Game termination

Games end only under the ordinary rules of chess, including the fifty-move rule. There is no
ply cap, so tournament results are not artificially drawn from non-terminal positions.

Adjudication by material or by Stockfish was rejected: material adjudication rewards
grabbing material and shuffling, engine adjudication imports an outside judge into the match
protocol rather than into evaluation, and either breaks comparability with published runs.

### Schedule and resume

The schedule is a pure function of the sorted entry list and `games_per_pair`, so resume is a
set difference rather than an inference from history. Every scheduled game carries a stable
key of tournament, pairing, and game index, stored on the game row under a unique index.
A result write is therefore idempotent: a lost response costs one replayed game and cannot
create a duplicate.

Games are **interleaved** rather than pair-major — round one plays one game of every pairing,
then round two, and so on — with colors alternating by game index. Interruption is a
first-class case in this design, and an interleaved run stopped at any point leaves every
pairing with a comparable number of games and balanced colors. A pair-major run stopped
early yields complete data on some pairings and nothing on others.

Interleaving wants every model resident simultaneously. Sessions are held for the whole run
when total declared package bytes fit a configured memory budget, falling back to pair-major
ordering when they do not. The rules explicitly permit caching a session across games.

### Failures

Faults reported by the worker are submission faults and lose the game, as the rules already
required. Runner-side faults — closed tab, reload, sleep, failed write — are retryable and
cost nobody a point. Each scheduled game carries an attempt counter and becomes a recorded
forfeit after a fixed number of attempts, so a package that reliably crashes the runner
cannot stall the tournament forever.

### One runner, pinned

The tournament row stores `runner_id` (a UUID persisted in `localStorage`), a human-typed
`runner_label`, and captured metadata: user agent, `hardwareConcurrency`, `deviceMemory`,
platform, arena version. A heartbeat maintained while running prevents a second tab from
starting — which would not merely duplicate work but contend for CPU and silently corrupt
every move's time budget.

Resuming from a different `runner_id` is refused by default. An administrator may override,
because a tournament permanently stuck behind a dead laptop is worse than a slightly impure
one, but the override is appended to the tournament row and displayed beside the standings
forever.

### Verification

Registration is not a form submit. It loads the package through the existing
`loadBrowserModel()` path and plays a short smoke game, and only then marks the entry
verified. This catches a broken submission days early rather than on the morning, and it
places the burden on the person who submitted it.

No runner-side pre-flight is performed. Time limits are set by the administrator when
configuring the tournament.

### Ranking

Standings are total points, with wins, draws, losses, score percentage and games played.
With a balanced round robin, raw points are correct and a rating fit would add nothing.
Ties share the title.

The results page also shows distinct games per pairing. Standings stand regardless, but a
pairing that produced two distinct games out of two hundred should be visibly different from
one that produced two hundred.

### Stockfish

Stockfish plays no part in the tournament pipeline. It would consume more compute than the
tournament itself and, if run concurrently, would steal CPU from a package that is on the
clock. Existing behaviour is unchanged: opening a finished game analyses it on demand.

### Spectator broadcasts

Every tournament game opens an ephemeral public live-game record before its first move.
After a move's wall-clock measurement has ended, the runner awaits a bounded snapshot write
before starting the opponent's clock. Spectator work therefore never overlaps package
inference, and a failed broadcast write degrades the spectator view without changing or
stopping the game. Completed rows in `games` remain the only source for standings.

The same live-game interface supports ordinary arena games when their player opts into an
unlisted watch link. Both paths publish a full, revisioned SAN history rather than an event
delta, so a spectator can recover after missing any number of polls. Spectators poll public
read-only state and reconstruct the position locally; they never connect to the pinned runner
or claim its lease. A finished broadcast resolves to the permanent recorded game under the
same game id.

## Code structure

The game loop is extracted from `site/app/arena/arena-client.tsx` into a React-free function:

```ts
playGame(white, black, { moveTimeLimitMs, seed }) => GameResult
```

It drives `chess.js`, calls `model.predict()`, enforces the per-move timeout, and returns the
result, termination, PGN and per-move timings. The arena calls it for a single game; the
tournament runner calls it in a loop. One implementation of the rules, two front ends, and
the component that decides outcomes becomes directly unit-testable.

The runner page holds resident sessions, the schedule, a progress bar, live standings, and a
per-game write awaited before the next game begins. Persistent write failure stops the run
rather than playing games that are not being recorded.

## Data model

`tournaments` carries status (`registration` → `running` → `completed`), the configuration
(`games_per_pair`, `move_time_limit_ms`, memory budget), runner identity,
metadata, heartbeat, and recorded runner changes.

`tournament_entries` links a tournament to a registered model version and its owning human
player, with verification state and smoke-test measurements. Entries are editable by their
owner only while the tournament is in registration.

`games` gains `tournament_id`, pairing, and game index, with a unique index across them.
Tournament games are ordinary games: they appear in history, on player pages, and can be
opened and analysed like any other.

`live_games` stores one short-lived, revisioned snapshot per broadcast game, including its
players, SAN history, phase, and optional tournament slot. Publisher credentials are stored
only as hashes, snapshots expire after one day, and no live row contributes to a result.

## Permissions

Administrators are an email allowlist checked against the ChatGPT-authenticated user; they
create tournaments and move them between phases. Any signed-in user may register, and may
edit only their own entry, only during registration. The runner page requires a signed-in
user but not an administrator, so the link can be handed to whoever's laptop is being
borrowed.

Entries are visible to everyone throughout. There are no openings left to reveal, so the
information a competitor gains from seeing another's entry early is limited to the reference
itself.
