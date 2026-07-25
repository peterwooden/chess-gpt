const sources = [
  {
    title: "Neural Networks: Zero to Hero",
    author: "Andrej Karpathy",
    href: "https://karpathy.ai/zero-to-hero.html",
  },
  {
    title: "Understanding Deep Learning",
    author: "Simon J. D. Prince",
    href: "https://udlbook.github.io/udlbook/",
  },
  {
    title: "Speech and Language Processing",
    author: "Dan Jurafsky & James H. Martin",
    href: "https://web.stanford.edu/~jurafsky/slp3/ed3book.pdf",
  },
];

export default function Home() {
  return (
    <main>
      <header className="hero">
        <p className="eyebrow">Chess GPT learning lab</p>
        <h1>Build.<br />Measure.<br />Explain.</h1>
        <p className="hero-copy">
          A first-principles course in machine learning, built around one concrete mission: train a small chess language model that can win.
        </p>
        <span className="status">Curriculum co-design in progress</span>
      </header>

      <section className="principles" aria-labelledby="principles-title">
        <p className="eyebrow">The learning contract</p>
        <h2 id="principles-title">Every concept must earn its place.</h2>
        <div className="principle-grid">
          <article>
            <span>01</span>
            <h3>Motivation first</h3>
            <p>Start with a chess-model problem that the new idea helps us solve.</p>
          </article>
          <article>
            <span>02</span>
            <h3>Mechanism second</h3>
            <p>Build the smallest inspectable version before relying on abstractions.</p>
          </article>
          <article>
            <span>03</span>
            <h3>Evidence always</h3>
            <p>Predict, measure, inspect failures, and explain what changed.</p>
          </article>
        </div>
      </section>

      <section className="sources" aria-labelledby="sources-title">
        <p className="eyebrow">Primary sources</p>
        <h2 id="sources-title">A course with foundations.</h2>
        <p className="section-copy">The final chapter sequence will be chosen with you, then each short lesson will draw from these deeper sources.</p>
        <div className="source-list">
          {sources.map((source) => (
            <a href={source.href} key={source.href}>
              <span><strong>{source.title}</strong><small>{source.author}</small></span>
              <i aria-hidden="true">↗</i>
            </a>
          ))}
        </div>
      </section>

      <footer>One focused hour per week · one trustworthy experiment at a time</footer>
    </main>
  );
}
