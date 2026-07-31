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

### No opening book

Every game starts from the standard position. Openings are part of chess, and the tournament
should measure them.

The cost is understood and accepted. Under this protocol a pairing of two deterministic
packages produces exactly two distinct games, one per color, and every further game is a
replay. Sample size within a pairing therefore depends entirely on whether a competitor
chose to make their submission stochastic. The harness applies no special treatment for
this: duplicate games are played, recorded, and scored like any others.

This also invalidated the previous tiebreak rule, which resolved ties by playing more
opening pairs. Against deterministic packages that rule cannot terminate. Ties now share
the title.

Note for future work: `adapters/board-policy/entry.source.js` selects by `argmax` and never
calls the supplied `random()`, so both currently published candidates are deterministic.

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

A tournament configures `max_plies`; reaching it is a draw with termination `max_plies`.
This matches `src/chess_gpt/snapshot_match.py` exactly, so tournament results stay directly
comparable with the paired-match results already published in this repository. The default is
200 plies, or 100 moves each.

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

## Code structure

The game loop is extracted from `site/app/arena/arena-client.tsx` into a React-free function:

```ts
playGame(white, black, { moveTimeLimitMs, maxPlies, seed }) => GameResult
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
(`games_per_pair`, `move_time_limit_ms`, `max_plies`, memory budget), runner identity,
metadata, heartbeat, and recorded runner changes.

`tournament_entries` links a tournament to a registered model version and its owning human
player, with verification state and smoke-test measurements. Entries are editable by their
owner only while the tournament is in registration.

`games` gains `tournament_id`, pairing, and game index, with a unique index across them.
Tournament games are ordinary games: they appear in history, on player pages, and can be
opened and analysed like any other.

## Permissions

Administrators are an email allowlist checked against the ChatGPT-authenticated user; they
create tournaments and move them between phases. Any signed-in user may register, and may
edit only their own entry, only during registration. The runner page requires a signed-in
user but not an administrator, so the link can be handed to whoever's laptop is being
borrowed.

Entries are visible to everyone throughout. There are no openings left to reveal, so the
information a competitor gains from seeing another's entry early is limited to the reference
itself.
