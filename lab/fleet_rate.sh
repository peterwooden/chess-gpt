#!/bin/bash
# Pipelined rating queue: waits for each fleet checkpoint, rates it on the ladder.
set -u
cd "$(dirname "$0")/.."
OUT=runs/lab/fleet

R() {
  id=$1; shift
  ckpt=$OUT/$id.pt
  n=0; until [ -f "$ckpt" ] || [ -f "$OUT/$id.SKIP" ] || [ $n -ge 960 ]; do sleep 30; n=$((n+1)); done
  if [ ! -f "$ckpt" ]; then echo "MISSING $id"; return; fi
  if [ -f "$OUT/$id.rating.json" ]; then echo "skip $id"; return; fi
  echo "== rate $id =="
  uv run python -m lab.ladder rate --checkpoint "$ckpt" "$@" 2>/dev/null | tail -1
}

for id in 00-control 01-winner1600 02-winner2000 03-decisive 04-nobullet 05-20k 06-40k \
          07-dedup 08-both1800 09-drawsonly 10-hist2 11-hist8 12-repetition 13-material \
          14-nostate 15-mlp 16-persquare 17-moe 18-geobias 19-pieceinit 20-d64L2 21-d96L4 \
          22-d160L8 23-heads8 24-heads2 25-ff2 26-dropout 27-value025 28-value05 29-value10 \
          30-auxmat 31-auxplies 32-labelsmooth 34-epochs3 35-cosine 36-lr3e3 37-b4096 \
          38-warmup 39-ft-elite 40-ft-decisive 41-selfplay 42-expertiter 43-distill; do
  R "$id"
done
# 33 plays through its value head — greedy policy would rate an untrained head
R 33-valueonly --search value1 --tag ""
# --- inference-layer experiments: re-rate existing checkpoints under new decodes ---
uv run python -m lab.ladder rate --checkpoint $OUT/00-control.pt --temperature 0.7 --tag temp07 2>/dev/null | tail -1
uv run python -m lab.ladder rate --checkpoint $OUT/00-control.pt --temperature 1.0 --top-k 3 --tag top3 2>/dev/null | tail -1
uv run python -m lab.ladder rate --checkpoint $OUT/28-value05.pt --search value1 --tag search1 2>/dev/null | tail -1
uv run python -m lab.ladder rate --checkpoint $OUT/28-value05.pt --search value1 --contempt 0.15 --tag contempt 2>/dev/null | tail -1
echo FLEET-RATE-DONE
