"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";

import SplitSimulator from "./split-simulator";

const STORAGE_KEY = "chess-gpt:chapter-1-mission-1-v2";
const COMPLETION_CODE = "CGPT-C1M1-GAMEHASH";
const PINNED_BASELINE =
  "https://github.com/peterwooden/chess-gpt/blob/ebe8aa257fdda759056df8c28ba15afeed18d57d/src/chess_gpt/baseline.py#L107-L109";

type SavedProgress = {
  forecast: number | null;
  confidence: number | null;
  forecastChecked: boolean;
  implementation: number | null;
  implementationChecked: boolean;
  retrieval: Record<string, number>;
  retrievalChecked: boolean;
  retrievalAttempts: number;
};

const initialProgress: SavedProgress = {
  forecast: null,
  confidence: null,
  forecastChecked: false,
  implementation: null,
  implementationChecked: false,
  retrieval: {},
  retrievalChecked: false,
  retrievalAttempts: 0,
};

const exampleRows = [
  { id: "R1", context: "(game start)", target: "e4" },
  { id: "R2", context: "e4", target: "e5" },
  { id: "R3", context: "e4 e5", target: "Nf3" },
  { id: "R4", context: "e4 e5 Nf3", target: "Nc6" },
  { id: "R5", context: "e4 e5 Nf3 Nc6", target: "Bb5" },
  { id: "R6", context: "e4 e5 Nf3 Nc6 Bb5", target: "a6" },
];

const implementations = [
  {
    label: "A",
    title: "Reserve the last rows",
    code: "cut = round(len(games) * 0.9)\ntrain = games[:cut]\nvalidation = games[cut:]",
    note: "Games stay intact, but membership depends on row order: re-sort the file and the split changes, and ordered files often hide chronology or source bias.",
  },
  {
    label: "B",
    title: "Shuffle expanded positions",
    code: "examples = expand_to_positions(games)\nshuffle(examples)\ntrain, validation = split(examples, 0.9)",
    note: "The shuffle looks rigorous, but rows from one game can land on both sides of the boundary — the exact leak this mission is about.",
  },
  {
    label: "C",
    title: "Hash stable game identity",
    code: "digest = blake2b(f\"{seed}:{game.site}\".encode(), digest_size=8).digest()\nbucket = int.from_bytes(digest) % 100\nsplit = \"validation\" if bucket < 10 else \"train\"",
    note: "The whole game follows one deterministic assignment that depends only on its identity, not on row order.",
  },
];

const retrievalQuestions = [
  {
    id: "unit",
    prompt: "What is the split unit for our current evaluation?",
    choices: ["One SAN token", "One chess game", "One opening name"],
    correct: 1,
    hints: [
      "Ask which rows share information with each other. Rows generated from the same game certainly do.",
      "The split unit is the smallest thing that must stay whole. For us that is an entire game — every one of its rows goes to the same side.",
    ],
  },
  {
    id: "hash",
    prompt: "Why hash the stable game ID with a seed?",
    choices: [
      "To make the model's moves legal",
      "To reduce the model's size",
      "To reproduce the same assignment regardless of row order",
    ],
    correct: 2,
    hints: [
      "Think about what happens to a row-order-based split the day someone re-sorts or re-downloads the file.",
      "Hashing a stable identity makes the assignment a property of the game itself, so any machine, any file order, any day produces the same split.",
    ],
  },
  {
    id: "size",
    prompt: "On a fixed dataset, what does a larger validation split usually buy and cost?",
    choices: [
      "A steadier estimate, at the cost of fewer training games",
      "A lower loss, at no data cost",
      "More parameters, at the cost of a noisier estimate",
    ],
    correct: 0,
    hints: [
      "The dataset is a fixed pool. Every game you give to validation has to come from somewhere.",
      "More validation games average away more noise, so the estimate steadies — but the training set shrinks by exactly those games.",
    ],
  },
];

export default function LessonClient() {
  const [progress, setProgress] = useState<SavedProgress>(initialProgress);
  const [hydrated, setHydrated] = useState(false);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    let saved = initialProgress;
    try {
      if (raw) saved = { ...initialProgress, ...(JSON.parse(raw) as SavedProgress) };
    } catch {
      window.localStorage.removeItem(STORAGE_KEY);
    }
    const timer = window.setTimeout(() => {
      setProgress(saved);
      setHydrated(true);
    }, 0);
    return () => window.clearTimeout(timer);
  }, []);

  useEffect(() => {
    if (hydrated) window.localStorage.setItem(STORAGE_KEY, JSON.stringify(progress));
  }, [hydrated, progress]);

  const forecastCorrect = progress.forecastChecked && progress.forecast === 2;
  const implementationCorrect = progress.implementationChecked && progress.implementation === 2;
  const retrievalCorrect = useMemo(
    () => retrievalQuestions.every((question) => progress.retrieval[question.id] === question.correct),
    [progress.retrieval],
  );
  const complete = forecastCorrect && implementationCorrect && progress.retrievalChecked && retrievalCorrect;

  function reset() {
    setProgress(initialProgress);
    setCopied(false);
    window.localStorage.removeItem(STORAGE_KEY);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  async function copyCode() {
    await navigator.clipboard.writeText(COMPLETION_CODE);
    setCopied(true);
  }

  function checkRetrieval() {
    setProgress((current) => ({
      ...current,
      retrievalChecked: true,
      retrievalAttempts: retrievalQuestions.every(
        (question) => current.retrieval[question.id] === question.correct,
      )
        ? current.retrievalAttempts
        : current.retrievalAttempts + 1,
    }));
  }

  return (
    <main className="lesson-page">
      <nav className="lesson-nav" aria-label="Lesson navigation">
        <Link href="/">CGPT / LAB</Link>
        <div className="lesson-nav-context"><span>Chapter 01 · Mission 01</span><Link href="/models">Models</Link><Link href="/arena">Arena</Link></div>
      </nav>

      <header className="lesson-hero">
        <div>
          <p className="eyebrow">≈ 25 minutes · hook → predict → explain → play → check</p>
          <h1>Split games,<br />not positions.</h1>
          <p className="lede">
            Every decision in this course — architecture, learning rate, data size — will be judged by one
            number: validation accuracy. This mission is about making sure that number tells the truth.
          </p>
        </div>
        <aside className="lesson-brief">
          <span>Mission question</span>
          <strong>Could the model have seen this game before?</strong>
          <p>If the answer is yes, your “held-out” score measures recognition, not chess.</p>
        </aside>
      </header>

      <section className="lesson-section" aria-labelledby="hook-title">
        <div className="lesson-section-heading">
          <p className="eyebrow">00 · Why this matters</p>
          <h2 id="hook-title">A score that lies politely.</h2>
        </div>
        <div className="lesson-prose">
          <p>
            Our count-based baseline currently predicts the next human move in <strong>22.4%</strong> of
            held-out positions. Suppose that next month a shiny new neural model reports <strong>78%</strong>{" "}
            validation accuracy. Champagne? Not yet — because there is a boring way to reach 78% without
            learning any more chess: let the evaluation quietly reuse games the model already studied.
          </p>
          <p>
            That failure mode is called{" "}
            <a href="/glossary#data-leakage">data leakage</a>, and it is not a beginner’s mistake — leaky
            evaluations regularly slip into published research. For us it is also the single most expensive
            mistake available, because every later experiment chooses between models by comparing validation
            numbers. If those numbers reward memorization, we will spend the tournament’s 100 MB and one
            exaFLOP optimizing for the wrong thing, and discover it on match night.
          </p>
          <p>
            The plan: commit to a prediction, see the exact mechanism of the leak with rows you can point at,
            watch it inflate a score in a simulator, and then approve the split our real code uses.
          </p>
        </div>
      </section>

      <section className="lesson-section" aria-labelledby="forecast-title">
        <div className="lesson-section-heading">
          <p className="eyebrow">01 · Predict before you read</p>
          <h2 id="forecast-title">Predict the direction of the error.</h2>
          <p>
            Here is the setup. We expand each game into overlapping next-move examples, shuffle all examples
            from every game together, then take 10% as validation. Positions from one game can land on both
            sides of the boundary.
          </p>
        </div>
        <article className="lesson-task">
          <h3>Relative to genuinely unseen games, apparent validation accuracy will usually be…</h3>
          <div className="lesson-choice-grid" role="radiogroup" aria-label="Forecast validation accuracy">
            {["Lower than on new games", "About the same", "Higher than on new games"].map((choice, index) => (
              <button
                type="button"
                role="radio"
                aria-checked={progress.forecast === index}
                className={progress.forecast === index ? "lesson-choice selected" : "lesson-choice"}
                onClick={() => setProgress((current) => ({ ...current, forecast: index, forecastChecked: false }))}
                key={choice}
              ><span>{String.fromCharCode(65 + index)}</span><strong>{choice}</strong></button>
            ))}
          </div>
          <fieldset className="confidence">
            <legend>Confidence in this prediction</legend>
            {["30%", "60%", "90%"].map((label, index) => (
              <button
                type="button"
                className={progress.confidence === index ? "confidence-button selected" : "confidence-button"}
                aria-pressed={progress.confidence === index}
                onClick={() => setProgress((current) => ({ ...current, confidence: index }))}
                key={label}
              >{label}</button>
            ))}
          </fieldset>
          <button
            className="lesson-submit"
            type="button"
            disabled={progress.forecast === null || progress.confidence === null}
            onClick={() => setProgress((current) => ({ ...current, forecastChecked: true }))}
          >Commit prediction</button>
          {progress.forecastChecked ? (
            <div className={forecastCorrect ? "lesson-feedback correct" : "lesson-feedback incorrect"} aria-live="polite">
              <strong>{forecastCorrect ? "Direction supported: higher." : "Not quite — here is a hint."}</strong>
              <p>{forecastCorrect
                ? "Overlapping rows let validation reward recognition of games already represented in training, so the score flatters the model. Now read on — the next section shows the exact mechanism, with rows you can point at."
                : "Ask which protocol hands the model an easier exam: one on games it partly studied, or one on games it has never seen? Read the next section, then come back and change your answer — that is allowed and encouraged."}</p>
            </div>
          ) : null}
        </article>
      </section>

      <section className="lesson-section" aria-labelledby="mechanism-title">
        <div className="lesson-section-heading">
          <p className="eyebrow">02 · The mechanism</p>
          <h2 id="mechanism-title">The exam answers are hiding in the study guide.</h2>
        </div>

        <div className="lesson-prose">
          <h3>What a validation score is supposed to measure</h3>
          <p>
            Training accuracy answers an easy question: “how well did the model fit the games it studied?”
            With enough capacity, memorizing wins that contest without any understanding. The question we
            actually care about is different: <em>“how will it move in a game nobody has played yet?”</em> —
            because that is literally what the tournament is. A{" "}
            <a href="/glossary#validation-split">validation set</a> is a stand-in for that future: games we
            possess but the model has never touched. The score it produces is trustworthy only while “never
            touched” stays literally true.
          </p>

          <h3>One game becomes many training rows</h3>
          <p>
            Language models don’t train on games; they train on <strong>(context → next move)</strong> rows.
            Here is a six-ply Ruy Lopez opening and every row it generates:
          </p>
        </div>

        <div className="table-scroll">
          <table className="rows-table">
            <thead>
              <tr><th>Row</th><th>Context the model sees</th><th>Target it must predict</th></tr>
            </thead>
            <tbody>
              {exampleRows.map((row) => (
                <tr key={row.id} className={row.id === "R4" ? "row-highlight" : undefined}>
                  <th scope="row">{row.id}</th>
                  <td><code>{row.context}</code></td>
                  <td><code>{row.target}</code></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="lesson-prose">
          <p>
            Notice the redundancy: R6’s context contains R2, R3, R4, and R5 in their entirety. These six rows
            are not six independent facts about chess — they are <em>one game photographed from six angles</em>.
          </p>

          <h3>Hold out one row, leak it through its neighbours</h3>
          <p>
            Say the shuffle sends R4 to validation and the other five rows to training. The exam question is:
            “after <code>e4 e5 Nf3</code>, what came next?” The answer is <code>Nc6</code>. Now look at what
            the model studied. Training row R5 begins <code>e4 e5 Nf3 Nc6…</code> — the exam’s answer is
            printed, verbatim, inside a study row. So is R6. A model with spare capacity doesn’t need to
            understand knight development; it can simply remember “that game where <code>Nc6</code> followed{" "}
            <code>Nf3</code>” and recite it back.
          </p>
        </div>

        <figure className="leak-diagram" aria-label="Diagram: the held-out row R4 is answered by training rows R5 and R6, whose contexts contain the answer Nc6">
          <svg viewBox="0 0 720 268" role="img" aria-hidden="true">
            <text x={16} y={26} fontSize={13} fontWeight={800} fill="#687068" letterSpacing={1.5}>TRAINING</text>
            <text x={452} y={26} fontSize={13} fontWeight={800} fill="#687068" letterSpacing={1.5}>VALIDATION</text>

            {[
              { y: 42, label: "R2", text: "e4 → e5" },
              { y: 84, label: "R3", text: "e4 e5 → Nf3" },
              { y: 126, label: "R5", text: "e4 e5 Nf3 Nc6 → Bb5", leak: true },
              { y: 168, label: "R6", text: "e4 e5 Nf3 Nc6 Bb5 → a6", leak: true },
            ].map((row) => (
              <g key={row.label}>
                <rect x={16} y={row.y} width={396} height={32} rx={4} fill="#fffdf7" stroke={row.leak ? "#b85b29" : "#c9c2b2"} strokeWidth={row.leak ? 2 : 1} />
                <text x={30} y={row.y + 21} fontSize={12.5} fontWeight={800} fill="#b85b29" fontFamily="ui-monospace, monospace">{row.label}</text>
                <text x={66} y={row.y + 21} fontSize={13} fill="#17211b" fontFamily="ui-monospace, monospace">{row.text}</text>
              </g>
            ))}

            <rect x={139} y={130} width={35} height={24} rx={4} fill="none" stroke="#b85b29" strokeWidth={2} strokeDasharray="4 3" />
            <rect x={139} y={172} width={35} height={24} rx={4} fill="none" stroke="#b85b29" strokeWidth={2} strokeDasharray="4 3" />
            <text x={156} y={252} textAnchor="middle" fontSize={12} fontWeight={700} fill="#b85b29">the held-out answer, sitting in training</text>

            <rect x={452} y={84} width={252} height={44} rx={4} fill="#dce7df" stroke="#143f31" strokeWidth={2} />
            <text x={466} y={103} fontSize={12.5} fontWeight={800} fill="#143f31" fontFamily="ui-monospace, monospace">R4 · held out</text>
            <text x={466} y={120} fontSize={13} fill="#17211b" fontFamily="ui-monospace, monospace">e4 e5 Nf3 → Nc6?</text>

            <path d="M 177 142 C 310 142 365 108 446 106" fill="none" stroke="#b85b29" strokeWidth={2} strokeDasharray="6 4" markerEnd="url(#leak-arrow)" />
            <defs>
              <marker id="leak-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
                <path d="M0,0 L8,4 L0,8 z" fill="#b85b29" />
              </marker>
            </defs>
          </svg>
          <figcaption>
            The held-out question R4 is answered verbatim inside training rows R5 and R6: their contexts
            contain <code>Nf3 Nc6</code>, the exact transition validation is supposed to test.
          </figcaption>
        </figure>

        <div className="lesson-prose">
          <p>
            To be fair: a <em>different</em> game that reached <code>e4 e5 Nf3</code> and continued{" "}
            <code>Nc6</code> would also “give away” this answer — but that is legitimate generalization, a
            pattern learned across many players. The leak is that this exact game answers its own exam. And
            the distinction we care about — pattern versus memory — is precisely the distinction the leaky
            split erases.
          </p>

          <h3>Randomness is not independence</h3>
          <p>
            The shuffle in the leaky protocol is genuinely random, and that is the trap: randomness controls{" "}
            <em>which</em> rows cross the boundary, not <em>whether related rows may cross it</em>. The fix is
            choosing the right <a href="/glossary#split-unit">split unit</a> — the smallest thing that must
            stay whole. Rows from one game share information, so the unit is the game. The same reasoning
            appears everywhere in machine learning: split by patient in medical imaging, by speaker in audio,
            by day in market data. Whenever examples share a source, the source is the unit.
          </p>

          <h3>Three lines that keep the evidence honest</h3>
          <p>Our baseline decides each game’s fate with one tiny function:</p>
        </div>

        <pre className="code-walk"><code>{`def is_validation_game(site: str, seed: int, validation_percent: int) -> bool:
    digest = hashlib.blake2b(f"{seed}:{site}".encode(), digest_size=8).digest()
    return int.from_bytes(digest) % 100 < validation_percent`}</code></pre>

        <div className="lesson-prose">
          <ul className="code-walk-list">
            <li><code>site</code> is the game’s Lichess URL — a stable identity that never changes, no matter how the file is sorted or re-downloaded.</li>
            <li><code>blake2b(...)</code> turns <code>seed + identity</code> into a huge, effectively random — but perfectly deterministic — number.</li>
            <li><code>% 100</code> drops that number into one of a hundred buckets; <code>&lt; validation_percent</code> sends, say, buckets 0–9 to validation. That is a 10% split.</li>
          </ul>
          <p>
            Four properties fall out of those three lines: the whole game lands on one side (assignment
            depends only on identity), the split is <em>reproducible</em> on any machine in any row order,
            no chronology bias can sneak in, and changing the <code>seed</code> deals a completely fresh
            split without touching the data.{" "}
            <a href={PINNED_BASELINE}>Read the real function</a> — pinned to the exact commit, it is these
            same three lines.
          </p>

          <h3>What honesty costs</h3>
          <p>
            A game-level split scores lower — that is the point, it stopped grading memory — and its estimate
            wobbles more when validation holds only a few games, because whole games are coarser coins to flip.
            The rough rule: quadrupling the number of validation games halves the wobble. How much data to
            spend on a steadier estimate is a genuine trade-off, and it is Mission 4’s topic. You can feel it
            first, in the simulator below.
          </p>
        </div>
      </section>

      <section className="lesson-section" aria-labelledby="simulator-title">
        <div className="lesson-section-heading">
          <p className="eyebrow">03 · See it happen</p>
          <h2 id="simulator-title">Watch the leak inflate a score.</h2>
          <p>
            This simulator runs the whole story as a repeatable experiment: 200 synthetic games of 30
            positions each, played by a model whose true skill on new games is 30%, but which recalls about
            80% of the lines it studied. Each dot is one complete experiment — a fresh split, then an
            evaluation.
          </p>
        </div>

        <SplitSimulator />

        <div className="lesson-prose sim-notes">
          <p><strong>Four things to notice while you play:</strong></p>
          <ul>
            <li>The amber cloud sits far above the truth at every setting — it reports the model’s memory, not its chess. It drifts down as you hold out more of each game, because partial memorization leaks partially. Diluted leakage is still leakage.</li>
            <li>The green cloud brackets the dashed truth line. Lower, honest, useful.</li>
            <li>Drag validation share down to 5%: the green dots scatter, because a handful of validation games makes a noisy estimate. Push it to 50%: steadier — but watch the training-games counter fall.</li>
            <li>The leak is not noise, it is <em>bias</em>: no amount of redrawing pulls the amber cloud toward the truth.</li>
          </ul>
          <p className="sim-disclaimer">
            The toy’s assumptions (skill 30%, recall 80%, uniform games) are made up, but the shape survives
            any reasonable numbers. The Chapter 1 experiment measures the real effect on the real baseline.
          </p>
        </div>
      </section>

      <section className="lesson-section" aria-labelledby="compare-title">
        <div className="lesson-section-heading">
          <p className="eyebrow">04 · Implementation comparison</p>
          <h2 id="compare-title">Which split would you approve?</h2>
          <p>
            You are reviewing three pull requests. Each claims to split games 90/10. Choose the one that keeps
            the evidence trustworthy for our current dataset — and know why the other two fail.
          </p>
        </div>
        <div className="implementation-grid" role="radiogroup" aria-label="Data split implementations">
          {implementations.map((implementation, index) => (
            <button
              type="button"
              role="radio"
              aria-checked={progress.implementation === index}
              className={progress.implementation === index ? "implementation-card selected" : "implementation-card"}
              onClick={() => setProgress((current) => ({ ...current, implementation: index, implementationChecked: false }))}
              key={implementation.label}
            >
              <span>{implementation.label}</span>
              <h3>{implementation.title}</h3>
              <pre><code>{implementation.code}</code></pre>
              <p>{implementation.note}</p>
            </button>
          ))}
        </div>
        <button
          className="lesson-submit"
          type="button"
          disabled={progress.implementation === null}
          onClick={() => setProgress((current) => ({ ...current, implementationChecked: true }))}
        >Check implementation</button>
        {progress.implementationChecked ? (
          <div className={implementationCorrect ? "lesson-feedback correct" : "lesson-feedback incorrect"} aria-live="polite">
            <strong>{implementationCorrect ? "Approve C — with one explicit assumption." : "Do not approve this one yet."}</strong>
            <p>{implementationCorrect
              ? "A stable hash of seed + game identity keeps the whole game together and makes assignment independent of row order. The stated assumption: site must be unique and stable per game — which is exactly what the source assignment below asks you to verify."
              : progress.implementation === 0
                ? "A keeps games intact, but re-sorting the file silently changes the split, and ordered files often carry chronology or source bias into one side. Look for assignment that depends on the game itself, not its position in a file."
                : "B is the leaky protocol from this lesson: rows from one game cross the boundary, and validation starts grading memory. Randomness cannot repair the wrong split unit."}</p>
          </div>
        ) : null}
      </section>

      <section className="lesson-section" aria-labelledby="retrieval-title">
        <div className="lesson-section-heading">
          <p className="eyebrow">05 · Retrieval checkpoint</p>
          <h2 id="retrieval-title">Close the loop without looking back.</h2>
          <p>
            All three must be correct at the same time. Wrong answers unlock hints — retry as often as you
            like; there is no penalty.
          </p>
        </div>
        <div className="retrieval-stack">
          {retrievalQuestions.map((question, questionIndex) => {
            const answer = progress.retrieval[question.id];
            const missed = progress.retrievalChecked && answer !== question.correct;
            const hint = missed
              ? question.hints[Math.min(progress.retrievalAttempts, question.hints.length) - 1] ?? null
              : null;
            return (
              <article className="retrieval-card" key={question.id}>
                <span>R{questionIndex + 1}</span>
                <h3>{question.prompt}</h3>
                <div className="retrieval-choices" role="radiogroup" aria-label={question.prompt}>
                  {question.choices.map((choice, choiceIndex) => (
                    <button
                      type="button"
                      role="radio"
                      aria-checked={answer === choiceIndex}
                      className={answer === choiceIndex ? "selected" : ""}
                      onClick={() => setProgress((current) => ({
                        ...current,
                        retrieval: { ...current.retrieval, [question.id]: choiceIndex },
                        retrievalChecked: false,
                      }))}
                      key={choice}
                    >{choice}</button>
                  ))}
                </div>
                {hint ? (
                  <div className="hint-box" aria-live="polite">
                    <strong>Hint {Math.min(progress.retrievalAttempts, question.hints.length)}</strong>
                    <p>{hint}</p>
                  </div>
                ) : null}
              </article>
            );
          })}
        </div>
        <button
          className="lesson-submit"
          type="button"
          disabled={Object.keys(progress.retrieval).length !== retrievalQuestions.length}
          onClick={checkRetrieval}
        >Check retrieval</button>
        {progress.retrievalChecked ? (
          <div className={retrievalCorrect ? "lesson-feedback correct" : "lesson-feedback incorrect"} aria-live="polite">
            <strong>{retrievalCorrect ? "All three mechanisms retrieved." : "At least one needs another look — hints are open above."}</strong>
            <p>{retrievalCorrect
              ? "The game is the split unit; stable identity makes assignment reproducible; more validation data steadies the estimate while shrinking the training pool."
              : "Change the answers marked with a hint and check again. Unlimited retries, no penalty — the goal is retrieval, not a score."}</p>
          </div>
        ) : null}
      </section>

      <section className={complete ? "mission-complete unlocked" : "mission-complete"} aria-live="polite">
        <p className="eyebrow">Mission handoff</p>
        {complete ? (
          <>
            <h2>Now teach it back.</h2>
            <p>Paste the code into Codex, then explain in your own words why position-level splitting changes apparent validation accuracy. That explanation — not this page view — is the mastery evidence.</p>
            <div className="mission-code"><code>{COMPLETION_CODE}</code><button type="button" onClick={copyCode}>{copied ? "Copied" : "Copy code"}</button></div>
          </>
        ) : (
          <>
            <h2>The code is still sealed.</h2>
            <p>Commit the forecast, approve the trustworthy implementation, and answer all three retrieval questions correctly.</p>
          </>
        )}
        <button className="lesson-reset" type="button" onClick={reset}>Reset this mission</button>
      </section>

      <section className="lesson-section sources" aria-labelledby="deeper-title">
        <div className="lesson-section-heading">
          <p className="eyebrow">06 · Go deeper · pick what you need</p>
          <h2 id="deeper-title">The trail beyond this page.</h2>
          <p>
            The source assignment for this mission (20–40 minutes): read the real split function and the
            pinned dataset manifest, then answer — <em>what identity must remain stable for our split to
            remain reproducible?</em> The rest of the list is optional depth, each entry labelled with what
            it gives you.
          </p>
        </div>
        <div className="source-list">
          <a href={PINNED_BASELINE}>
            <span><strong>is_validation_game — the real three lines</strong><small>Project evidence · pinned to the exact commit this lesson describes</small></span><i>↗</i>
          </a>
          <a href="https://github.com/peterwooden/chess-gpt/blob/ebe8aa257fdda759056df8c28ba15afeed18d57d/docs/BASELINE_0001.md">
            <span><strong>Baseline record 0001</strong><small>Project evidence · where the 22.4% held-out accuracy comes from</small></span><i>↗</i>
          </a>
          <a href="https://mlu-explain.github.io/train-test-validation/">
            <span><strong>MLU-Explain: Train, Test, and Validation Sets</strong><small>Interactive essay · the same ground with different visuals, ~10 minutes — ideal spaced repetition</small></span><i>↗</i>
          </a>
          <a href="https://www.kaggle.com/code/alexisbcook/data-leakage">
            <span><strong>Kaggle: Data Leakage</strong><small>Tutorial · leakage beyond splits — target leakage and contamination, with real competition disasters</small></span><i>↗</i>
          </a>
          <a href="https://www.deeplearningbook.org/contents/ml.html">
            <span><strong>Deep Learning, §§5.2–5.3</strong><small>Textbook · the formal version: generalization error and the i.i.d. assumption our split protects</small></span><i>↗</i>
          </a>
          <a href="https://www.ijcai.org/Proceedings/95-2/Papers/016.pdf">
            <span><strong>Kohavi 1995: Cross-Validation and Bootstrap</strong><small>Primary source · why k-fold exists, §§1–3 — optional until the Chapter 1 experiment</small></span><i>↗</i>
          </a>
          <Link href="/glossary">
            <span><strong>Course glossary</strong><small>Reference · every term this lesson used, with this project’s exact meanings</small></span><i>↗</i>
          </Link>
        </div>
      </section>

      <footer className="lesson-footer">
        <span>Chess GPT Learning Lab</span>
        <Link href="/">Return to roadmap</Link>
      </footer>
    </main>
  );
}
