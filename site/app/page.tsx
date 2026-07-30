"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";

const STORAGE_KEY = "chess-gpt:diagnostic-v1";

type Domain = "code" | "probability" | "optimization" | "experiments" | "tensors";

type DiagnosticQuestion = {
  id: string;
  domain: Domain;
  label: string;
  prompt: string;
  code?: string;
  choices: [string, string, string];
  correct: number;
  rationale: string;
};

const questions: DiagnosticQuestion[] = [
  {
    id: "q1",
    domain: "code",
    label: "Read a loop",
    prompt: "What is the final value of best?",
    code: "scores = [2, 5, 1]\nbest = scores[0]\nfor score in scores:\n    if score > best:\n        best = score",
    choices: ["It becomes two", "It becomes five", "It becomes one"],
    correct: 1,
    rationale: "The loop replaces best only when it sees a larger value, so it retains the maximum: 5.",
  },
  {
    id: "q2",
    domain: "probability",
    label: "Read a loss",
    prompt: "The played move was Nc6. Model A assigned it probability 0.8; Model B assigned 0.2. Which model has lower negative log-likelihood?",
    choices: ["Model B has lower loss", "Both models have equal loss", "Model A has lower loss"],
    correct: 2,
    rationale: "Negative log-likelihood decreases as the probability assigned to the observed target increases.",
  },
  {
    id: "q3",
    domain: "optimization",
    label: "Follow a gradient",
    prompt: "For L(w) = (w − 3)² at w = 1, the derivative is −4. A small gradient-descent step moves w in which direction?",
    choices: ["It moves toward zero", "It moves toward three", "It stays exactly one"],
    correct: 1,
    rationale: "Gradient descent subtracts the derivative. Subtracting a negative value increases w, toward the minimum at 3.",
  },
  {
    id: "q4",
    domain: "optimization",
    label: "Change a lever",
    prompt: "The learning rate becomes 100× larger and updates repeatedly overshoot the minimum. What is the most likely training behaviour?",
    choices: ["Loss oscillates or diverges repeatedly", "Loss decreases more smoothly throughout", "Model capacity becomes exactly smaller"],
    correct: 0,
    rationale: "A step size that is too large can jump across a useful minimum repeatedly or send parameters away from it.",
  },
  {
    id: "q5",
    domain: "experiments",
    label: "Change a split",
    prompt: "A fixed dataset changes from 10% to 40% validation data. What trade-off should you predict?",
    choices: [
      "Validation noisier; training pool larger",
      "Validation unchanged; training data free",
      "Validation steadier; training pool smaller",
    ],
    correct: 2,
    rationale: "More validation examples usually reduce metric uncertainty, but a fixed dataset then leaves fewer examples for learning.",
  },
  {
    id: "q6",
    domain: "experiments",
    label: "Change capacity",
    prompt: "Model width increases substantially while data and training stay fixed. What is the safest initial prediction?",
    choices: [
      "Training loss rises; overfitting risk falls",
      "Training loss falls; overfitting risk rises",
      "Both losses remain exactly unchanged",
    ],
    correct: 1,
    rationale: "More capacity usually makes training data easier to fit, while generalization may worsen without enough data or regularization.",
  },
  {
    id: "q7",
    domain: "tensors",
    label: "Read a shape",
    prompt: "A batch has 32 games, 16 moves of context, and 64 values per move embedding. What is the resulting tensor shape?",
    choices: ["32 × 16 × 64", "16 × 32 × 64", "32 × 64 × 16"],
    correct: 0,
    rationale: "The axes preserve the stated order: batch, sequence position, then embedding features.",
  },
  {
    id: "q8",
    domain: "experiments",
    label: "Choose a protocol",
    prompt: "Which workflow preserves an honest final estimate while still allowing model selection?",
    choices: [
      "Validation updates; training selects; test waits",
      "Train updates; test selects; validation waits",
      "Train updates; validation selects; test waits",
    ],
    correct: 2,
    rationale: "Training data updates parameters, validation guides iteration, and the test set waits for infrequent final evaluation.",
  },
];

const chapters = [
  { number: "01", title: "What does it mean to learn?", motive: "Separate reusable patterns from memorized games.", concepts: "Data · targets · splits · baselines" },
  { number: "02", title: "How can a number learn?", motive: "Turn prediction errors into useful parameter changes.", concepts: "Hill climbing · loss · gradients" },
  { number: "03", title: "How do moves become probabilities?", motive: "Generalize beyond histories seen word-for-word.", concepts: "Tensors · embeddings · softmax" },
  { number: "04", title: "Did it learn or memorize?", motive: "Know whether improvement will survive unseen games.", concepts: "Capacity · overfitting · regularization" },
  { number: "05", title: "How fast should it learn?", motive: "Use the budget quickly without destabilizing training.", concepts: "Learning rate · batches · AdamW" },
  { number: "06", title: "How can any earlier move matter?", motive: "Reach relevant context anywhere in the game history.", concepts: "Attention · position · causal masks" },
  { number: "07", title: "What makes a Transformer trainable?", motive: "Assemble a stable model within the size limit.", concepts: "Residuals · normalization · depth" },
  { number: "08", title: "How should we spend the budget?", motive: "Allocate 100 MB and one exaFLOP deliberately.", concepts: "Precision · FLOPs · scaling" },
  { number: "09", title: "Does lower loss mean stronger chess?", motive: "Select for wins, not attractive training curves.", concepts: "Calibration · matches · Elo" },
  { number: "10", title: "How does an idea become evidence?", motive: "Make creative experiments credible and useful.", concepts: "Hypotheses · ablations · reproducibility" },
];

const rlChapters = [
  { number: "R1", title: "When is prediction the wrong objective?", motive: "Separate imitating human moves from maximizing game result.", concepts: "Policy · reward · return" },
  { number: "R2", title: "How does a win teach earlier moves?", motive: "Assign credit when the clearest signal arrives at checkmate.", concepts: "Value · Bellman · Monte Carlo · TD" },
  { number: "R3", title: "Should we learn values, actions, or both?", motive: "Choose an algorithm family that fits chess and the budget.", concepts: "Policy gradients · actor-critic · exploration" },
  { number: "R4", title: "Can self-play improve us honestly?", motive: "Control opponent drift, variance, compute, and evaluation bias.", concepts: "Self-play · search · paired matches" },
];

const confidenceLabels = ["Guessing", "Some confidence", "Very confident"];

type SavedDiagnostic = {
  answers: Record<string, number>;
  confidence: Record<string, number>;
  submitted: boolean;
};

function diagnosticResult(answers: Record<string, number>, confidence: Record<string, number>) {
  const correct = questions.map((question) => answers[question.id] === question.correct);
  const score = correct.filter(Boolean).length;
  const correctMask = correct.reduce((mask, value, index) => mask | (value ? 1 << index : 0), 0);
  const highConfidenceMisses = questions.filter(
    (question, index) => !correct[index] && confidence[question.id] === 2,
  ).length;
  const gaps = [...new Set(questions.filter((_, index) => !correct[index]).map((question) => question.domain))];
  const track = score >= 7 ? "direct" : score >= 4 ? "bridge" : "foundations";
  const title = {
    direct: "Ready for the direct path",
    bridge: "A short foundations bridge will help",
    foundations: "Start with the foundations path",
  }[track];
  const code = `CGPT-D0-${score}-${correctMask.toString(16).padStart(2, "0").toUpperCase()}-H${highConfidenceMisses}`;
  return { code, correct, gaps, highConfidenceMisses, score, title, track };
}

export default function Home() {
  const [answers, setAnswers] = useState<Record<string, number>>({});
  const [confidence, setConfidence] = useState<Record<string, number>>({});
  const [submitted, setSubmitted] = useState(false);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return;
    try {
      const saved = JSON.parse(raw) as SavedDiagnostic;
      const timer = window.setTimeout(() => {
        setAnswers(saved.answers ?? {});
        setConfidence(saved.confidence ?? {});
        setSubmitted(Boolean(saved.submitted));
      }, 0);
      return () => window.clearTimeout(timer);
    } catch {
      window.localStorage.removeItem(STORAGE_KEY);
    }
  }, []);

  useEffect(() => {
    const saved: SavedDiagnostic = { answers, confidence, submitted };
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(saved));
  }, [answers, confidence, submitted]);

  const answered = Object.keys(answers).length;
  const confidenceSet = Object.keys(confidence).length;
  const ready = answered === questions.length && confidenceSet === questions.length;
  const result = useMemo(
    () => (submitted ? diagnosticResult(answers, confidence) : null),
    [answers, confidence, submitted],
  );

  function choose(questionId: string, choice: number) {
    if (submitted) return;
    setAnswers((current) => ({ ...current, [questionId]: choice }));
  }

  function chooseConfidence(questionId: string, value: number) {
    if (submitted) return;
    setConfidence((current) => ({ ...current, [questionId]: value }));
  }

  function submitDiagnostic() {
    if (!ready) return;
    setSubmitted(true);
    window.setTimeout(() => document.getElementById("diagnostic-result")?.scrollIntoView({ behavior: "smooth" }), 0);
  }

  function resetDiagnostic() {
    setAnswers({});
    setConfidence({});
    setSubmitted(false);
    setCopied(false);
    window.localStorage.removeItem(STORAGE_KEY);
    document.getElementById("diagnostic")?.scrollIntoView({ behavior: "smooth" });
  }

  async function copyCode() {
    if (!result) return;
    await navigator.clipboard.writeText(result.code);
    setCopied(true);
  }

  return (
    <main>
      <nav className="masthead" aria-label="Course status">
        <a href="#top" className="wordmark">CGPT / LAB</a>
        <div className="masthead-status">
          <span>Placement diagnostic</span>
          <div className="progress-track" aria-hidden="true"><i style={{ width: `${(answered / questions.length) * 100}%` }} /></div>
          <strong>{answered}/{questions.length}</strong>
        </div>
      </nav>

      <header className="hero" id="top">
        <div className="hero-copy">
          <p className="eyebrow">Chess GPT learning lab · Field notebook 00</p>
          <h1>Predict before<br />you train.</h1>
          <p className="lede">Learn machine learning by forecasting what each lever will do, testing it on a real chess model, and explaining the evidence.</p>
          <div className="hero-actions">
            <Link className="primary-link" href="/chapter-1/data-splits">Continue Chapter 1 <span>→</span></Link>
            <a className="text-link" href="#diagnostic">Review the diagnostic</a>
            <Link className="text-link" href="/arena">Open the browser arena</Link>
            <Link className="text-link" href="/models">Discover arena models</Link>
            <Link className="text-link" href="/history">Browse player histories</Link>
            <Link className="text-link" href="/glossary">Browse the glossary</Link>
            <a className="text-link" href="#roadmap">Inspect the roadmap</a>
          </div>
        </div>
        <aside className="field-card" aria-label="Course facts">
          <span className="field-card-label">Course parameters</span>
          <dl>
            <div><dt>Time</dt><dd>5 hrs / week</dd></div>
            <div><dt>Path</dt><dd>10 core + 4 RL</dd></div>
            <div><dt>Loop</dt><dd>Toy → chess</dd></div>
            <div><dt>Goal</dt><dd>Win + understand</dd></div>
          </dl>
        </aside>
      </header>

      <section className="contract section-shell" aria-labelledby="contract-title">
        <div className="section-heading">
          <p className="eyebrow">The learning contract</p>
          <h2 id="contract-title">Understanding has observable evidence.</h2>
        </div>
        <div className="contract-grid">
          <article><span>01</span><h3>Forecast</h3><p>State direction, magnitude, assumptions, confidence, and likely failure modes.</p></article>
          <article><span>02</span><h3>Discriminate</h3><p>Choose the best implementation and explain why the alternatives fail.</p></article>
          <article><span>03</span><h3>Measure</h3><p>Run one controlled change against a trustworthy baseline.</p></article>
          <article><span>04</span><h3>Update</h3><p>Explain surprises from memory and revise the mental model.</p></article>
        </div>
      </section>

      <section className="roadmap section-shell" id="roadmap" aria-labelledby="roadmap-title">
        <div className="section-heading roadmap-heading">
          <div>
            <p className="eyebrow">The complete path</p>
            <h2 id="roadmap-title">Ten causal questions.</h2>
          </div>
          <p>Ten chapters form the core. A four-part reinforcement-learning extension follows once the supervised model and evaluation protocol are trustworthy.</p>
        </div>
        <div className="chapter-list">
          {chapters.map((chapter, index) => (
            <article className={index === 0 ? "chapter chapter-active" : "chapter"} key={chapter.number}>
              <div className="chapter-check" aria-label={`Chapter ${chapter.number} ${index === 0 ? "in progress" : "locked"}`}>{index === 0 ? "→" : "□"}</div>
              <span className="chapter-number">{chapter.number}</span>
              <div className="chapter-copy"><h3>{index === 0 ? <Link href="/chapter-1/data-splits">{chapter.title}</Link> : chapter.title}</h3><p>{chapter.motive}</p></div>
              <small>{chapter.concepts}</small>
              {index === 0 ? <Link className="chapter-open" href="/chapter-1/data-splits">Mission 1 →</Link> : <span className="locked">Locked</span>}
            </article>
          ))}
        </div>
        <div className="extension-block">
          <div className="extension-heading">
            <div>
              <p className="eyebrow">Phase II · optional competitive extension</p>
              <h3>Reinforcement learning</h3>
            </div>
            <p>RL asks a different question: not “what move did a human play?” but “which action improves expected game result?” Self-play only reaches the tournament model after the shared-data rule is clarified.</p>
          </div>
          <div className="chapter-list extension-list">
            {rlChapters.map((chapter) => (
              <article className="chapter" key={chapter.number}>
                <div className="chapter-check" aria-label={`Extension ${chapter.number} locked`}>□</div>
                <span className="chapter-number">{chapter.number}</span>
                <div className="chapter-copy"><h3>{chapter.title}</h3><p>{chapter.motive}</p></div>
                <small>{chapter.concepts}</small>
                <span className="locked">Locked</span>
              </article>
            ))}
          </div>
        </div>
      </section>

      <section className="diagnostic section-shell" id="diagnostic" aria-labelledby="diagnostic-title">
        <div className="diagnostic-intro">
          <p className="eyebrow">Placement diagnostic · 10–15 minutes</p>
          <h2 id="diagnostic-title">Show me how you reason today.</h2>
          <p>No preparation and no pass mark. Answer without searching, record your confidence, then submit once. Your result determines the starting path for Chapter 1.</p>
          <div className="diagnostic-rule"><strong>Important:</strong> a confident wrong answer is more useful for teaching than a disguised guess.</div>
        </div>

        <div className="question-stack">
          {questions.map((question, questionIndex) => (
            <article className="question-card" key={question.id}>
              <header>
                <span>Q{String(questionIndex + 1).padStart(2, "0")}</span>
                <small>{question.label}</small>
              </header>
              <h3>{question.prompt}</h3>
              {question.code ? <pre><code>{question.code}</code></pre> : null}
              <div className="choices" role="radiogroup" aria-label={question.prompt}>
                {question.choices.map((choice, choiceIndex) => (
                  <button
                    type="button"
                    role="radio"
                    aria-checked={answers[question.id] === choiceIndex}
                    className={answers[question.id] === choiceIndex ? "choice selected" : "choice"}
                    disabled={submitted}
                    onClick={() => choose(question.id, choiceIndex)}
                    key={choice}
                  >
                    <span>{String.fromCharCode(65 + choiceIndex)}</span>
                    <strong>{choice}</strong>
                  </button>
                ))}
              </div>
              <fieldset className="confidence" disabled={submitted}>
                <legend>How confident are you?</legend>
                {confidenceLabels.map((label, confidenceIndex) => (
                  <button
                    type="button"
                    className={confidence[question.id] === confidenceIndex ? "confidence-button selected" : "confidence-button"}
                    aria-pressed={confidence[question.id] === confidenceIndex}
                    onClick={() => chooseConfidence(question.id, confidenceIndex)}
                    key={label}
                  >{label}</button>
                ))}
              </fieldset>
              {result ? (
                <div className={result.correct[questionIndex] ? "rationale correct" : "rationale incorrect"}>
                  <strong>{result.correct[questionIndex] ? "Prediction supported" : `Best answer: ${question.choices[question.correct]}`}</strong>
                  <p>{question.rationale}</p>
                </div>
              ) : null}
            </article>
          ))}
        </div>

        {!result ? (
          <div className="submit-panel">
            <div><span>{answered} answers · {confidenceSet} confidence ratings</span><strong>{ready ? "Ready to submit" : `${questions.length - Math.min(answered, confidenceSet)} decisions remain`}</strong></div>
            <button type="button" disabled={!ready} onClick={submitDiagnostic}>Submit diagnostic</button>
          </div>
        ) : (
          <section className="result-panel" id="diagnostic-result" aria-live="polite">
            <p className="eyebrow">Diagnostic complete · {result.score}/{questions.length}</p>
            <h2>{result.title}</h2>
            <p>
              {result.gaps.length === 0
                ? "No prerequisite gaps appeared in this small sample."
                : `The first adaptive missions should probe: ${result.gaps.join(", ")}.`}
              {result.highConfidenceMisses > 0 ? ` ${result.highConfidenceMisses} confident miss${result.highConfidenceMisses === 1 ? "" : "es"} will receive special attention.` : " Your confidence was well calibrated on incorrect answers."}
            </p>
            <div className="handoff-code"><span>Paste this into Codex</span><code>{result.code}</code></div>
            <div className="result-actions">
              <button type="button" onClick={copyCode}>{copied ? "Code copied" : "Copy diagnostic code"}</button>
              <button className="reset-button" type="button" onClick={resetDiagnostic}>Reset diagnostic</button>
            </div>
            <small>Your teacher uses this code and one follow-up prediction to choose the Chapter 1 path. Mission progress is stored separately on this device.</small>
          </section>
        )}
      </section>

      <section className="sources section-shell" aria-labelledby="sources-title">
        <div className="section-heading">
          <p className="eyebrow">The source shelf</p>
          <h2 id="sources-title">Primary sources, assigned with purpose.</h2>
        </div>
        <div className="source-list">
          <a href="https://karpathy.ai/zero-to-hero.html"><span><strong>Neural Networks: Zero to Hero</strong><small>Andrej Karpathy · build-from-scratch spine</small></span><i>↗</i></a>
          <a href="https://udlbook.github.io/udlbook/"><span><strong>Understanding Deep Learning</strong><small>Simon J. D. Prince · visual textbook spine</small></span><i>↗</i></a>
          <a href="https://web.stanford.edu/~jurafsky/slp3/ed3book.pdf"><span><strong>Speech and Language Processing</strong><small>Jurafsky & Martin · language-model theory</small></span><i>↗</i></a>
          <a href="https://mml-book.github.io/"><span><strong>Mathematics for Machine Learning</strong><small>Deisenroth, Faisal & Ong · selective mathematics</small></span><i>↗</i></a>
          <a href="https://www.deeplearningbook.org/"><span><strong>Deep Learning</strong><small>Goodfellow, Bengio & Courville · selective theory</small></span><i>↗</i></a>
          <a href="https://spinningup.openai.com/en/latest/"><span><strong>Spinning Up in Deep RL</strong><small>OpenAI · conceptual RL extension</small></span><i>↗</i></a>
        </div>
      </section>

      <footer className="footer">
        <span>Chess GPT Learning Lab</span>
        <p>Five hours a week · one trustworthy experiment at a time</p>
      </footer>
    </main>
  );
}
