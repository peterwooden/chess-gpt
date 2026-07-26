"use client";

import { useMemo, useState } from "react";

// Deterministic PRNG so server-rendered SVG matches the client hydration exactly.
function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const GAMES = 200;
const POSITIONS_PER_GAME = 30;
const TRUE_SKILL = 0.3;
const RECALL = 0.8;
const RESAMPLES = 40;
const BASE_SEED = 20260726;

const LEAKY_COLOR = "#b85b29";
const HONEST_COLOR = "#2f8f63";

type Resample = { leakyAccuracy: number; honestAccuracy: number };

type SimulationResult = {
  resamples: Resample[];
  leakyMean: number;
  honestMean: number;
  honestSpread: number;
  trainingGames: number;
  validationGames: number;
  validationPositions: number;
};

function clamp(value: number, low: number, high: number) {
  return Math.min(high, Math.max(low, value));
}

function simulate(validationPercent: number, seed: number): SimulationResult {
  const share = validationPercent / 100;
  const validationGames = Math.round(GAMES * share);
  const resamples: Resample[] = [];

  for (let r = 0; r < RESAMPLES; r += 1) {
    const random = mulberry32(seed + r * 7919);

    // Every game gets its own true difficulty, so estimates from few games wobble.
    const freshSkill: number[] = [];
    const recallSkill: number[] = [];
    for (let g = 0; g < GAMES; g += 1) {
      freshSkill.push(clamp(TRUE_SKILL + (random() - 0.5) * 0.24, 0.05, 0.6));
      recallSkill.push(clamp(RECALL + (random() - 0.5) * 0.1, 0.6, 0.95));
    }

    // Game-level split: shuffle game indices, take the first slice as validation.
    const order = Array.from({ length: GAMES }, (_, index) => index);
    for (let i = order.length - 1; i > 0; i -= 1) {
      const j = Math.floor(random() * (i + 1));
      [order[i], order[j]] = [order[j], order[i]];
    }
    let honestCorrect = 0;
    for (let v = 0; v < validationGames; v += 1) {
      const game = order[v];
      for (let p = 0; p < POSITIONS_PER_GAME; p += 1) {
        if (random() < freshSkill[game]) honestCorrect += 1;
      }
    }
    const honestTotal = validationGames * POSITIONS_PER_GAME;

    // Position-level split: each position flips its own coin, so almost every
    // validation position comes from a game that is also (partly) in training.
    let leakyCorrect = 0;
    let leakyTotal = 0;
    for (let g = 0; g < GAMES; g += 1) {
      let heldOut = 0;
      for (let p = 0; p < POSITIONS_PER_GAME; p += 1) {
        if (random() < share) heldOut += 1;
      }
      const seenFraction = (POSITIONS_PER_GAME - heldOut) / POSITIONS_PER_GAME;
      const effective = freshSkill[g] + (recallSkill[g] - freshSkill[g]) * seenFraction;
      for (let p = 0; p < heldOut; p += 1) {
        if (random() < effective) leakyCorrect += 1;
      }
      leakyTotal += heldOut;
    }

    resamples.push({
      leakyAccuracy: leakyTotal === 0 ? 0 : (leakyCorrect / leakyTotal) * 100,
      honestAccuracy: honestTotal === 0 ? 0 : (honestCorrect / honestTotal) * 100,
    });
  }

  const leakyMean = resamples.reduce((sum, item) => sum + item.leakyAccuracy, 0) / RESAMPLES;
  const honestMean = resamples.reduce((sum, item) => sum + item.honestAccuracy, 0) / RESAMPLES;
  const honestVariance =
    resamples.reduce((sum, item) => sum + (item.honestAccuracy - honestMean) ** 2, 0) / RESAMPLES;

  return {
    resamples,
    leakyMean,
    honestMean,
    honestSpread: Math.sqrt(honestVariance),
    trainingGames: GAMES - validationGames,
    validationGames,
    validationPositions: validationGames * POSITIONS_PER_GAME,
  };
}

const WIDTH = 720;
const HEIGHT = 300;
const PLOT_LEFT = 16;
const PLOT_RIGHT = WIDTH - 16;
const AXIS_Y = HEIGHT - 34;
const ROWS = [
  { key: "leaky" as const, label: "Split by position — the leaky protocol", color: LEAKY_COLOR, top: 62, bottom: 118 },
  { key: "honest" as const, label: "Split by game — the honest protocol", color: HONEST_COLOR, top: 168, bottom: 224 },
];

function xFor(accuracy: number) {
  return PLOT_LEFT + (accuracy / 100) * (PLOT_RIGHT - PLOT_LEFT);
}

export default function SplitSimulator() {
  const [validationPercent, setValidationPercent] = useState(10);
  const [round, setRound] = useState(0);

  const result = useMemo(
    () => simulate(validationPercent, BASE_SEED + round * 104729),
    [validationPercent, round],
  );

  const jitter = useMemo(() => {
    const random = mulberry32(BASE_SEED + round * 104729 + 13);
    return Array.from({ length: RESAMPLES }, () => random());
  }, [round]);

  const truthX = xFor(TRUE_SKILL * 100);

  return (
    <figure className="simulator" aria-label="Split simulator comparing leaky and honest validation protocols">
      <div className="sim-controls">
        <label>
          <span>
            Validation share <strong>{validationPercent}%</strong>
          </span>
          <input
            type="range"
            min={5}
            max={50}
            step={5}
            value={validationPercent}
            onChange={(event) => setValidationPercent(Number(event.target.value))}
            aria-label="Validation share percent"
          />
        </label>
        <button type="button" onClick={() => setRound((current) => current + 1)}>
          Redraw {RESAMPLES} experiments
        </button>
      </div>

      <div className="sim-legend" aria-hidden="true">
        <span><i style={{ background: LEAKY_COLOR }} /> split by position</span>
        <span><i style={{ background: HONEST_COLOR }} /> split by game</span>
        <span><i className="sim-legend-truth" /> true skill on new games</span>
      </div>

      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        role="img"
        aria-label={`Dot plot of ${RESAMPLES} simulated experiments. Position-level splits measure about ${Math.round(result.leakyMean)} percent while the model's true skill on new games is 30 percent. Game-level splits measure about ${Math.round(result.honestMean)} percent.`}
      >
        {[0, 25, 50, 75, 100].map((tick) => (
          <g key={tick}>
            <line x1={xFor(tick)} y1={40} x2={xFor(tick)} y2={AXIS_Y} stroke="#dcd6c6" strokeWidth={1} />
            <text x={xFor(tick)} y={AXIS_Y + 18} textAnchor="middle" fontSize={11} fill="#687068">
              {tick}%
            </text>
          </g>
        ))}
        <line x1={PLOT_LEFT} y1={AXIS_Y} x2={PLOT_RIGHT} y2={AXIS_Y} stroke="#c9c2b2" strokeWidth={1} />
        <text x={PLOT_RIGHT} y={AXIS_Y + 32} textAnchor="end" fontSize={11} fill="#687068">
          measured validation accuracy
        </text>

        <line x1={truthX} y1={26} x2={truthX} y2={AXIS_Y} stroke="#17211b" strokeWidth={1.5} strokeDasharray="5 4" />
        <text x={truthX + 6} y={20} fontSize={11.5} fontWeight={700} fill="#17211b">
          true skill ≈ 30%
        </text>

        {ROWS.map((row) => {
          const mean = row.key === "leaky" ? result.leakyMean : result.honestMean;
          return (
            <g key={row.key}>
              <text x={PLOT_LEFT} y={row.top - 8} fontSize={12.5} fontWeight={750} fill={row.color}>
                {row.label}
              </text>
              {result.resamples.map((sample, index) => {
                const accuracy = row.key === "leaky" ? sample.leakyAccuracy : sample.honestAccuracy;
                const y = row.top + 8 + jitter[index] * (row.bottom - row.top - 16);
                return (
                  <circle key={index} cx={xFor(accuracy)} cy={y} r={4.5} fill={row.color} fillOpacity={0.5}>
                    <title>{`Experiment ${index + 1}: measured ${accuracy.toFixed(1)}%`}</title>
                  </circle>
                );
              })}
              <line x1={xFor(mean)} y1={row.top} x2={xFor(mean)} y2={row.bottom} stroke={row.color} strokeWidth={2} />
              <text
                x={xFor(mean)}
                y={row.bottom + 15}
                textAnchor="middle"
                fontSize={11.5}
                fontWeight={800}
                fill={row.color}
              >
                mean {mean.toFixed(0)}%
              </text>
            </g>
          );
        })}
      </svg>

      <figcaption className="sim-readout">
        <div>
          <span>Leaky split reports</span>
          <strong style={{ color: LEAKY_COLOR }}>{result.leakyMean.toFixed(1)}%</strong>
          <small>+{(result.leakyMean - TRUE_SKILL * 100).toFixed(0)} points of pure illusion</small>
        </div>
        <div>
          <span>Honest split reports</span>
          <strong style={{ color: HONEST_COLOR }}>{result.honestMean.toFixed(1)}%</strong>
          <small>± {result.honestSpread.toFixed(1)} points across redraws</small>
        </div>
        <div>
          <span>Training games</span>
          <strong>{result.trainingGames}</strong>
          <small>of {GAMES} available</small>
        </div>
        <div>
          <span>Validation games</span>
          <strong>{result.validationGames}</strong>
          <small>{result.validationPositions} positions</small>
        </div>
      </figcaption>
    </figure>
  );
}
