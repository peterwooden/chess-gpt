"""Cheap controlled test: does the search teach the judge anything outcomes don't?

Same 200k positions, same frozen-trunk fine-tune of champion f1's value head,
only the label differs: (A) own 1-ply search backup vs (B) game outcome.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import chess
import numpy as np
import torch

from chess_gpt.snapshot_model import encode_board
from lab.data import load_slice
from lab.match import _white_score, load_model

CHAMPION = Path("runs/lab/fleet/f1-warmcos-full.pt")
CASTLE_BITS = [chess.BB_H1, chess.BB_A1, chess.BB_H8, chess.BB_A8]  # K Q k q


def board_from_row(squares: np.ndarray, state: np.ndarray) -> chess.Board:
    board = chess.Board(None)
    for square, code in enumerate(squares):
        if code:
            board.set_piece_at(
                square, chess.Piece(int((code - 1) % 6) + 1, bool(code <= 6))
            )
    board.turn = chess.WHITE if state[0] == 0 else chess.BLACK
    rights = 0
    for flag, bit in zip(state[1:5], CASTLE_BITS):
        if flag:
            rights |= bit
    board.castling_rights = rights
    board.ep_square = int(state[5]) if state[5] < 64 else None
    board.halfmove_clock = int(state[6])
    board.fullmove_number = 20
    return board


def search_labels(model, data, indices, device, chunk=256) -> np.ndarray:
    labels = np.zeros(len(indices), dtype=np.float32)
    hf_all, ht_all = data.extras["history_from"], data.extras["history_to"]
    for start in range(0, len(indices), chunk):
        block = indices[start : start + chunk]
        succ_inputs, owners, terminal = [], [], {}
        mover_white = np.zeros(len(block), dtype=bool)
        for slot, row in enumerate(block):
            board = board_from_row(data.squares[row], data.state[row])
            mover_white[slot] = board.turn == chess.WHITE
            for move in board.legal_moves:
                board.push(move)
                if board.is_checkmate():
                    terminal.setdefault(slot, []).append(0.0 if board.turn == chess.WHITE else 1.0)
                elif board.is_game_over(claim_draw=True):
                    terminal.setdefault(slot, []).append(0.5)
                else:
                    snap = encode_board(board)
                    succ_inputs.append((
                        snap.squares, snap.state,
                        np.append(hf_all[row][1:], move.from_square),
                        np.append(ht_all[row][1:], move.to_square),
                    ))
                    owners.append(slot)
                board.pop()
        scores: dict[int, list[float]] = {k: list(v) for k, v in terminal.items()}
        if succ_inputs:
            tensors = [
                torch.tensor(np.array([s[i] for s in succ_inputs]), dtype=torch.long, device=device)
                for i in range(4)
            ]
            with torch.no_grad():
                white = _white_score(model(*tensors)["value"]).cpu().numpy()
            for owner, value in zip(owners, white):
                scores.setdefault(owner, []).append(float(value))
        for slot in range(len(block)):
            options = scores.get(slot, [0.5])
            labels[start + slot] = max(options) if mover_white[slot] else min(options)
    return labels


def finetune(model, data, indices, targets, device, epochs=3, lr=1e-3, batch=1024):
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.value.parameters():
        p.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.value.parameters(), lr=lr)
    squares = torch.from_numpy(data.squares[indices]).to(device)
    state = torch.from_numpy(data.state[indices]).to(device)
    hf = torch.from_numpy(data.extras["history_from"][indices]).to(device)
    ht = torch.from_numpy(data.extras["history_to"][indices]).to(device)
    target = torch.from_numpy(targets).to(device)
    total = epochs * ((len(indices) + batch - 1) // batch)
    step = 0
    model.train()
    rng = np.random.default_rng(0)
    for _ in range(epochs):
        order = torch.from_numpy(rng.permutation(len(indices))).to(device)
        for s in range(0, len(order), batch):
            b = order[s : s + batch]
            for g in optimizer.param_groups:
                g["lr"] = lr * 0.5 * (1 + np.cos(np.pi * step / total))
            out = model(squares[b].long(), state[b].long(), hf[b].long(), ht[b].long())
            loss = ((_white_score(out["value"]) - target[b]) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", type=int, default=200_000)
    parser.add_argument("--games", type=int, default=40_000)
    args = parser.parse_args()

    started = time.perf_counter()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    data = load_slice(Path("data/downloads/lab/enriched-240k.parquet"), args.games)
    rng = np.random.default_rng(20260801)
    indices = rng.choice(data.positions, size=min(args.positions, data.positions), replace=False)
    indices.sort()

    saved = torch.load(CHAMPION, weights_only=True, map_location="cpu")
    model = load_model(CHAMPION).to(device)
    print(json.dumps({"labeling": len(indices)}), flush=True)
    labels_a = search_labels(model, data, indices, device)
    labels_b = (1.0 - data.result[indices].astype(np.float32) / 2.0)
    outcome_scalar = labels_b
    disagreement = float(np.abs(labels_a - outcome_scalar).mean())
    print(json.dumps({"label_seconds": round(time.perf_counter() - started, 1),
                      "mean_label_disagreement": round(disagreement, 4)}), flush=True)

    for name, targets in (("A-searchlabel", labels_a), ("B-outcomelabel", labels_b)):
        arm = load_model(CHAMPION).to(device)
        finetune(arm, data, indices, targets, device)
        torch.save({"sweep_recipe": saved["sweep_recipe"], "config": {},
                    "model": {k: v.cpu() for k, v in arm.state_dict().items()}},
                   f"runs/lab/fleet/judge-{name}.pt")
        print(json.dumps({"trained": name}), flush=True)
    print(json.dumps({"total_seconds": round(time.perf_counter() - started, 1)}))


if __name__ == "__main__":
    main()
