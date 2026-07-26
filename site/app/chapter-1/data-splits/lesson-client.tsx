"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";

const STORAGE_KEY = "chess-gpt:chapter-1-mission-1-v1";
const COMPLETION_CODE = "CGPT-C1M1-GAMEHASH";

type SavedProgress = {
  forecast: number | null;
  confidence: number | null;
  forecastChecked: boolean;
  implementation: number | null;
  implementationChecked: boolean;
  retrieval: Record<string, number>;
  retrievalChecked: boolean;
};

const initialProgress: SavedProgress = {
  forecast: null,
  confidence: null,
  forecastChecked: false,
  implementation: null,
  implementationChecked: false,
  retrieval: {},
  retrievalChecked: false,
};

const implementations = [
  {
    label: "A",
    title: "Reserve the last rows",
    code: "cut = round(len(games) * 0.9)\ntrain = games[:cut]\nvalidation = games[cut:]",
    note: "Games stay intact, but assignment depends on row order and can inherit chronology or source bias.",
  },
  {
    label: "B",
    title: "Shuffle expanded positions",
    code: "examples = expand_to_positions(games)\nshuffle(examples)\ntrain, validation = split(examples, 0.9)",
    note: "The shuffle looks rigorous, but overlapping examples from one game can cross the boundary.",
  },
  {
    label: "C",
    title: "Hash stable game identity",
    code: "digest = blake2b(f\"{seed}:{game.site}\".encode(), digest_size=8).digest()\nbucket = int.from_bytes(digest) % 100\nsplit = \"validation\" if bucket < 10 else \"train\"",
    note: "The whole game follows one deterministic assignment independent of row order.",
  },
];

const retrievalQuestions = [
  {
    id: "unit",
    prompt: "What is the split unit for our current evaluation?",
    choices: ["One SAN token", "One chess game", "One opening name"],
    correct: 1,
  },
  {
    id: "hash",
    prompt: "Why hash the stable game ID with a seed?",
    choices: ["To make moves legal", "To reduce model size", "To reproduce assignment regardless of row order"],
    correct: 2,
  },
  {
    id: "size",
    prompt: "On a fixed dataset, what does a larger validation split usually buy and cost?",
    choices: ["Steadier estimate; fewer training games", "Lower loss; no data cost", "More parameters; noisier estimate"],
    correct: 0,
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

  return (
    <main className="lesson-page">
      <nav className="lesson-nav" aria-label="Lesson navigation">
        <Link href="/">CGPT / LAB</Link>
        <span>Chapter 01 · Mission 01</span>
      </nav>

      <header className="lesson-hero">
        <div>
          <p className="eyebrow">10–15 minutes · predict → discriminate → retrieve</p>
          <h1>Split games,<br />not positions.</h1>
          <p className="lede">A validation score is useful only when it measures the kind of novelty we claim it measures.</p>
        </div>
        <aside className="lesson-brief">
          <span>Mission question</span>
          <strong>Could the model recognize this game from training?</strong>
          <p>If yes, “held out” may be a label rather than a fact.</p>
        </aside>
      </header>

      <section className="lesson-section" aria-labelledby="forecast-title">
        <div className="lesson-section-heading">
          <p className="eyebrow">01 · Forecast before explanation</p>
          <h2 id="forecast-title">Predict the direction of the error.</h2>
          <p>We expand each game into overlapping next-move examples, shuffle all positions, then take 10% for validation. Positions from one game can land on both sides.</p>
        </div>
        <article className="lesson-task">
          <h3>Relative to genuinely unseen games, apparent validation accuracy will usually be…</h3>
          <div className="lesson-choice-grid" role="radiogroup" aria-label="Forecast validation accuracy">
            {["Lower", "About the same", "Higher"].map((choice, index) => (
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
              <strong>{forecastCorrect ? "Direction supported: higher" : "Try again: which split has an easier job?"}</strong>
              <p>{forecastCorrect
                ? "Overlapping prefixes let validation reward recognition of games already represented in training. The score becomes optimistic for the question “how well will this work on a new game?”"
                : "Compare what the model has already seen in training under the two protocols, then revise your answer."}</p>
            </div>
          ) : null}
        </article>
      </section>

      <section className="lesson-section" aria-labelledby="mechanism-title">
        <div className="lesson-section-heading">
          <p className="eyebrow">02 · See the mechanism</p>
          <h2 id="mechanism-title">Examples from one game share information.</h2>
        </div>
        <div className="game-envelope" aria-label="One chess game becomes overlapping examples that stay in one split">
          <div><span>Game A</span><strong>e4 e5 Nf3 Nc6</strong></div>
          <i>becomes</i>
          <div className="position-stack"><code>e4 → e5</code><code>e4 e5 → Nf3</code><code>e4 e5 Nf3 → Nc6</code></div>
          <i>stays in</i>
          <div className="split-stamp">TRAIN <em>or</em> VALIDATION</div>
        </div>
        <p className="lesson-note">The examples are not independent: they reuse the same moves and prefixes. Scattering them creates <a href="https://github.com/peterwooden/chess-gpt/blob/main/reference/glossary.html#data-leakage">data leakage</a>. The rule is about the <em>split unit</em>, not merely randomness.</p>
      </section>

      <section className="lesson-section" aria-labelledby="compare-title">
        <div className="lesson-section-heading">
          <p className="eyebrow">03 · Implementation comparison</p>
          <h2 id="compare-title">Which split would you approve?</h2>
          <p>Choose the implementation that keeps the evidence most trustworthy for our current dataset. “Random” is not automatically “independent.”</p>
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
            <strong>{implementationCorrect ? "Approve C—with one explicit assumption." : "Do not approve this one yet."}</strong>
            <p>{implementationCorrect
              ? "A stable hash of seed + game.site keeps the whole game together and makes assignment independent of row order. It depends on site being unique and stable."
              : progress.implementation === 0
                ? "A keeps games intact, but changing row order changes membership and ordered data can bias the two groups. Look for stable identity."
                : "B leaks overlapping positions from a single game across the boundary. Randomness cannot repair the wrong split unit."}</p>
          </div>
        ) : null}
      </section>

      <section className="lesson-section" aria-labelledby="retrieval-title">
        <div className="lesson-section-heading">
          <p className="eyebrow">04 · Retrieval checkpoint</p>
          <h2 id="retrieval-title">Close the loop without hints.</h2>
          <p>All three must be correct at the same time. Change an answer and retry as often as you need.</p>
        </div>
        <div className="retrieval-stack">
          {retrievalQuestions.map((question, questionIndex) => (
            <article className="retrieval-card" key={question.id}>
              <span>R{questionIndex + 1}</span>
              <h3>{question.prompt}</h3>
              <div className="retrieval-choices" role="radiogroup" aria-label={question.prompt}>
                {question.choices.map((choice, choiceIndex) => (
                  <button
                    type="button"
                    role="radio"
                    aria-checked={progress.retrieval[question.id] === choiceIndex}
                    className={progress.retrieval[question.id] === choiceIndex ? "selected" : ""}
                    onClick={() => setProgress((current) => ({
                      ...current,
                      retrieval: { ...current.retrieval, [question.id]: choiceIndex },
                      retrievalChecked: false,
                    }))}
                    key={choice}
                  >{choice}</button>
                ))}
              </div>
            </article>
          ))}
        </div>
        <button
          className="lesson-submit"
          type="button"
          disabled={Object.keys(progress.retrieval).length !== retrievalQuestions.length}
          onClick={() => setProgress((current) => ({ ...current, retrievalChecked: true }))}
        >Check retrieval</button>
        {progress.retrievalChecked ? (
          <div className={retrievalCorrect ? "lesson-feedback correct" : "lesson-feedback incorrect"} aria-live="polite">
            <strong>{retrievalCorrect ? "All three mechanisms retrieved." : "At least one mechanism needs another look."}</strong>
            <p>{retrievalCorrect
              ? "The game is the split unit; stable identity makes assignment reproducible; more validation data steadies the estimate while shrinking the training pool."
              : "Revisit what must stay together, what must remain stable, and what a fixed data pool forces you to trade."}</p>
          </div>
        ) : null}
      </section>

      <section className={complete ? "mission-complete unlocked" : "mission-complete"} aria-live="polite">
        <p className="eyebrow">Mission handoff</p>
        {complete ? (
          <>
            <h2>Now teach it back.</h2>
            <p>Paste the code into Codex, then explain in your own words why position-level splitting changes apparent validation accuracy. That explanation—not this page view—is the mastery evidence.</p>
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

      <section className="lesson-section sources" aria-labelledby="inspect-title">
        <div className="lesson-section-heading">
          <p className="eyebrow">Inspect the real project · 20–40 minute source assignment</p>
          <h2 id="inspect-title">Find the assumption in our code.</h2>
          <p>Read the actual function and answer: what identity must remain stable for this split to remain reproducible?</p>
        </div>
        <div className="source-list">
          <a href="https://github.com/peterwooden/chess-gpt/blob/main/src/chess_gpt/baseline.py#L43"><span><strong>GameRecord + is_validation_game</strong><small>Project evidence · inspect lines 43 and 107</small></span><i>↗</i></a>
          <a href="https://database.lichess.org/"><span><strong>Frozen dataset source</strong><small>Primary evidence · inspect monthly PGN files and checksums</small></span><i>↗</i></a>
          <a href="https://www.deeplearningbook.org/contents/ml.html"><span><strong>Deep Learning §§5.1–5.3</strong><small>Secondary explanation · generalization and partitions</small></span><i>↗</i></a>
        </div>
      </section>

      <footer className="lesson-footer">
        <span>Chess GPT Learning Lab</span>
        <Link href="/">Return to roadmap</Link>
      </footer>
    </main>
  );
}
