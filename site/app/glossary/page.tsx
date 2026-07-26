import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "Glossary · Chess GPT Learning Lab",
  description: "Short, canonical meanings for the terms the course uses repeatedly.",
};

type Term = { id: string; term: string; definition: string };

const groups: { heading: string; terms: Term[] }[] = [
  {
    heading: "Data and evaluation",
    terms: [
      {
        id: "training-example",
        term: "Training example",
        definition:
          "An input context paired with one or more target tokens whose prediction error contributes to the loss. It is not automatically the same thing as a whole chess game — one game usually generates many examples.",
      },
      {
        id: "split-unit",
        term: "Split unit",
        definition:
          "The smallest group assigned wholly to one data split. In our current evaluation it is a complete chess game, so overlapping positions from one game cannot cross from training into validation.",
      },
      {
        id: "data-leakage",
        term: "Data leakage",
        definition:
          "Information crosses into training or model selection that would not be available in the claimed evaluation scenario, making measured performance misleadingly optimistic. Splitting overlapping positions from one game across training and validation is one form.",
      },
      {
        id: "training-split",
        term: "Training split",
        definition: "The games whose targets may update model parameters.",
      },
      {
        id: "validation-split",
        term: "Validation split",
        definition:
          "Held-out games used repeatedly to choose models and hyperparameters. They never update parameters directly.",
      },
      {
        id: "test-split",
        term: "Test split",
        definition:
          "Held-out games used only for infrequent final evaluation after development decisions are made. Consulting it repeatedly to pick models quietly turns it into a second validation set.",
      },
      {
        id: "generalization",
        term: "Generalization",
        definition:
          "Performing well on examples the model never trained on, because it learned reusable patterns rather than the training data itself. The opposite failure — fitting the training data without the pattern — is memorization.",
      },
      {
        id: "baseline",
        term: "Baseline",
        definition:
          "A deliberately simple reference model measured under the same protocol as its challengers. An improvement claim means nothing except relative to a trustworthy baseline.",
      },
    ],
  },
  {
    heading: "Model mechanics",
    terms: [
      {
        id: "token",
        term: "Token",
        definition:
          "One discrete symbol from the vocabulary, represented inside the model by an integer ID. A chess move may require several tokens.",
      },
      {
        id: "vocabulary",
        term: "Vocabulary",
        definition:
          "The frozen, ordered set of all tokens the model can receive or predict. Its order defines the token IDs.",
      },
      {
        id: "parameter",
        term: "Parameter",
        definition:
          "A scalar value learned by gradient descent and stored in the submitted model. Under the proposed tournament rule, all unique trainable parameters count, including embeddings.",
      },
      {
        id: "logit",
        term: "Logit",
        definition: "One raw model score for a possible next token, before conversion into probabilities.",
      },
      {
        id: "legal-mask",
        term: "Legal-token mask",
        definition:
          "An inference rule that assigns zero probability to tokens that cannot continue any legal move. It constrains output; it does not prove the model learned chess legality.",
      },
    ],
  },
  {
    heading: "Experimental practice",
    terms: [
      {
        id: "experiment",
        term: "Experiment",
        definition: "A pre-stated hypothesis tested by one intentional change under otherwise comparable conditions.",
      },
      {
        id: "seed",
        term: "Random seed",
        definition:
          "A value that initializes a pseudorandom sequence. Recording it aids debugging and repetition, but cannot guarantee identical results across all software and hardware.",
      },
    ],
  },
];

export default function GlossaryPage() {
  return (
    <main className="lesson-page">
      <nav className="lesson-nav" aria-label="Glossary navigation">
        <Link href="/">CGPT / LAB</Link>
        <span>Reference · Glossary</span>
      </nav>

      <header className="lesson-hero">
        <div>
          <p className="eyebrow">Living reference</p>
          <h1>The words,<br />pinned down.</h1>
          <p className="lede">
            Short, canonical meanings for terms the course uses repeatedly. When ordinary ML usage is
            ambiguous, this project’s meaning wins.
          </p>
        </div>
      </header>

      {groups.map((group) => (
        <section className="lesson-section" key={group.heading} aria-label={group.heading}>
          <div className="lesson-section-heading">
            <p className="eyebrow">{group.heading}</p>
          </div>
          <dl className="glossary-list">
            {group.terms.map((entry) => (
              <div key={entry.id} id={entry.id} className="glossary-entry">
                <dt>
                  {entry.term} <a href={`#${entry.id}`} aria-label={`Link to ${entry.term}`}>#</a>
                </dt>
                <dd>{entry.definition}</dd>
              </div>
            ))}
          </dl>
        </section>
      ))}

      <footer className="lesson-footer">
        <span>Chess GPT Learning Lab</span>
        <Link href="/">Return to roadmap</Link>
      </footer>
    </main>
  );
}
