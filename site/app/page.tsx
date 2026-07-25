"use client";

import { useEffect, useMemo, useState } from "react";

const STORAGE_KEY = "chess-gpt:lesson-0001";
const COMPLETION_CODE = "CGPT-L1-TOKENS-5273";

type Question = {
  prompt: string;
  choices: string[];
  correct: number;
  success: string;
  retry: string;
  code?: boolean;
};

const questions: Question[] = [
  {
    prompt: "What is a token?",
    choices: [
      "A discrete symbol with an integer ID",
      "A complete game with a final result",
      "A board square with a fixed value",
    ],
    correct: 0,
    success:
      "Exactly. A token is one symbol from the frozen vocabulary, represented inside the model by an integer ID.",
    retry:
      "Not quite. Think smaller than a game or board: what single unit can the model receive or predict?",
  },
  {
    prompt: "Which packed sequence represents Raxe1#??",
    choices: [
      "<blunder> # x Rae1",
      "# <blunder> x Rae1",
      "<blunder> x # Rae1",
    ],
    correct: 0,
    success:
      "Correct. Quality comes first, then mate, capture, and finally the core rook move that completes the sequence.",
    retry:
      "Close. The model must see the quality signal before it predicts the move, and the core move must come last.",
    code: true,
  },
  {
    prompt: "Why can the same games create different learning problems?",
    choices: [
      "Their representations and targets can differ substantially",
      "Every training run finds identical model weights",
      "Legal masking removes every architectural difference automatically",
    ],
    correct: 0,
    success:
      "Yes. Token boundaries, context, targets, and loss masking decide what patterns the model is actually rewarded for learning.",
    retry:
      "Try again. The games are the raw events; the training pipeline decides what the model sees and what errors count.",
  },
];

function QuizCard({
  question,
  index,
  selected,
  passed,
  checked,
  onSelect,
  onCheck,
}: {
  question: Question;
  index: number;
  selected: number | null;
  passed: boolean;
  checked: boolean;
  onSelect: (choice: number) => void;
  onCheck: () => void;
}) {
  const feedback = passed ? question.success : checked ? question.retry : null;

  return (
    <section className={`quiz-card ${passed ? "quiz-card--passed" : ""}`}>
      <div className="quiz-card__heading">
        <span className="quiz-number">0{index + 1}</span>
        <span className="quiz-status">{passed ? "Mastered" : "Your turn"}</span>
      </div>
      <h3>{question.prompt}</h3>
      <div className="choices" role="radiogroup" aria-label={question.prompt}>
        {question.choices.map((choice, choiceIndex) => (
          <button
            className={`choice ${selected === choiceIndex ? "choice--selected" : ""}`}
            key={choice}
            type="button"
            role="radio"
            aria-checked={selected === choiceIndex}
            onClick={() => onSelect(choiceIndex)}
          >
            <span className="choice-marker" aria-hidden="true">
              {String.fromCharCode(65 + choiceIndex)}
            </span>
            {question.code ? <code>{choice}</code> : <span>{choice}</span>}
          </button>
        ))}
      </div>
      <button
        className="check-button"
        type="button"
        onClick={onCheck}
        disabled={selected === null || passed}
      >
        {passed ? "Answer locked" : "Check answer"}
      </button>
      <p className={`feedback ${passed ? "feedback--success" : "feedback--retry"}`} aria-live="polite">
        {feedback ?? "Choose an answer, then check your reasoning."}
      </p>
    </section>
  );
}

export default function Home() {
  const [selections, setSelections] = useState<(number | null)[]>([null, null, null]);
  const [passed, setPassed] = useState<boolean[]>([false, false, false]);
  const [checked, setChecked] = useState<boolean[]>([false, false, false]);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    const saved = window.localStorage.getItem(STORAGE_KEY);
    if (!saved) return;
    try {
      const state = JSON.parse(saved) as { selections?: (number | null)[]; passed?: boolean[] };
      if (state.selections?.length === questions.length) setSelections(state.selections);
      if (state.passed?.length === questions.length) setPassed(state.passed);
    } catch {
      window.localStorage.removeItem(STORAGE_KEY);
    }
  }, []);

  useEffect(() => {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify({ selections, passed }));
  }, [selections, passed]);

  const mastered = useMemo(() => passed.filter(Boolean).length, [passed]);
  const complete = mastered === questions.length;

  function selectAnswer(questionIndex: number, choiceIndex: number) {
    setSelections((current) => current.map((value, index) => (index === questionIndex ? choiceIndex : value)));
    setChecked((current) => current.map((value, index) => (index === questionIndex ? false : value)));
  }

  function checkAnswer(questionIndex: number) {
    setChecked((current) => current.map((value, index) => (index === questionIndex ? true : value)));
    if (selections[questionIndex] === questions[questionIndex].correct) {
      setPassed((current) => current.map((value, index) => (index === questionIndex ? true : value)));
    }
  }

  async function copyCode() {
    await navigator.clipboard.writeText(COMPLETION_CODE);
    setCopied(true);
  }

  function resetLesson() {
    setSelections([null, null, null]);
    setPassed([false, false, false]);
    setChecked([false, false, false]);
    setCopied(false);
    window.localStorage.removeItem(STORAGE_KEY);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  return (
    <main>
      <div className="progress-rail" aria-label={`${mastered} of ${questions.length} checks mastered`}>
        <div className="progress-rail__inner">
          <span>Lesson 0001</span>
          <div className="progress-track" aria-hidden="true">
            <span style={{ width: `${(mastered / questions.length) * 100}%` }} />
          </div>
          <strong>{mastered} / {questions.length}</strong>
        </div>
      </div>

      <header className="hero">
        <p className="eyebrow">Chess GPT field school · 12 minutes</p>
        <h1>One move.<br />Three representations.</h1>
        <p className="hero-copy">
          Trace a human chess move into the symbols a language model actually predicts—and prove you can do it without notes.
        </p>
        <a className="start-link" href="#lesson">Start the lesson <span aria-hidden="true">↓</span></a>
        <div className="move-stamp" aria-label="Example move exf8 equals queen check blunder">
          <span>human</span>
          <strong>exf8=Q+??</strong>
        </div>
      </header>

      <article id="lesson" className="lesson-shell">
        <section className="lesson-section">
          <p className="section-kicker">The central idea</p>
          <h2>The model never sees a chessboard.</h2>
          <p>
            It receives a sequence of integer IDs. Before we touch neural-network code, we need to understand how one human move becomes that sequence.
          </p>
          <div className="representation-flow">
            <div><span>Human notation</span><code>exf8=Q+??</code></div>
            <span className="flow-arrow" aria-hidden="true">→</span>
            <div><span>Readable pipeline</span><code>[??] [+] [x] [=Q] [ef8]</code></div>
            <span className="flow-arrow" aria-hidden="true">→</span>
            <div><span>Model vocabulary</span><code>&lt;blunder&gt; + x =Q ef8</code></div>
          </div>
          <p>
            The pipeline splits one event into quality, check, capture, promotion, and a core move. Packing maps each symbol to a frozen integer ID; the model predicts one next ID at a time.
          </p>
          <aside className="insight">
            <span>Why it matters</span>
            Two models can train on the same games yet solve different problems if their token boundaries, contexts, or rewarded targets differ.
          </aside>
        </section>

        <section className="lesson-section parameter-section">
          <p className="section-kicker">Tournament connection</p>
          <h2>The vocabulary consumes parameters.</h2>
          <p>At width 512, a vocabulary of 5,273 tokens needs this many learned embedding values:</p>
          <div className="equation"><span>5,273</span><i>×</i><span>512</span><i>=</i><strong>2,699,776</strong></div>
          <p>That is why our 50M cap includes embeddings: the representation is part of the model, not free scaffolding.</p>
        </section>

        <section className="quiz-section" aria-labelledby="quiz-title">
          <p className="section-kicker">Closed-book checkpoint</p>
          <h2 id="quiz-title">Earn your completion code.</h2>
          <p className="quiz-intro">Answer all three correctly. Wrong answers give a clue and remain retryable.</p>
          <div className="quiz-stack">
            {questions.map((question, index) => (
              <QuizCard
                key={question.prompt}
                question={question}
                index={index}
                selected={selections[index]}
                passed={passed[index]}
                checked={checked[index]}
                onSelect={(choice) => selectAnswer(index, choice)}
                onCheck={() => checkAnswer(index)}
              />
            ))}
          </div>

          {complete ? (
            <section className="completion" aria-live="polite">
              <p className="section-kicker">Lesson mastered</p>
              <h2>Your proof code</h2>
              <p>Paste this code into your Codex conversation. Your teacher will use it to record the lesson as learned.</p>
              <div className="completion-code"><code>{COMPLETION_CODE}</code></div>
              <button className="copy-button" type="button" onClick={copyCode}>{copied ? "Copied" : "Copy code"}</button>
            </section>
          ) : (
            <section className="completion completion--locked">
              <p className="section-kicker">Proof code locked</p>
              <h2>{questions.length - mastered} check{questions.length - mastered === 1 ? "" : "s"} remaining.</h2>
              <p>The code appears here when every concept check is mastered.</p>
            </section>
          )}
        </section>

        <section className="sources">
          <p className="section-kicker">Go deeper</p>
          <a href="https://huggingface.co/datasets/shazmate/lichess-chess-tokens">Shared dataset card <span>↗</span></a>
          <a href="https://karpathy.github.io/2019/04/25/recipe/">Karpathy’s training recipe <span>↗</span></a>
          <a href="https://github.com/karpathy/nn-zero-to-hero">Neural Networks: Zero to Hero <span>↗</span></a>
        </section>

        <footer>
          <p>Something unclear? Ask your Codex teacher before guessing.</p>
          <button type="button" onClick={resetLesson}>Reset lesson</button>
        </footer>
      </article>
    </main>
  );
}
