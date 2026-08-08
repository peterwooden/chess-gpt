"""Decode compact game-record parquets into the enriched rows lab/prepare.py materializes.

Input schema (one row per game): game_id fixed_size_binary(8), white_elo/black_elo
uint16, result int8 (prepare.py convention: 0 white won, 1 draw, 2 black won), time_base_s
uint16, time_inc_s uint8, termination int8, ply_count uint16, moves list<uint16>
with bits 0-5 from-square, 6-11 to-square, 12-14 promotion (0 none, 1 N, 2 B,
3 R, 4 Q). Square indexing matches python-chess (0=a1 .. 63=h8). Castling arrives
as the king's two-file move and en passant as the pawn's diagonal move to an empty
square; both are inferred during replay.

Output rows reproduce lab/prepare.py's encoding bit for bit (UNFLIPPED; the
trainer's flip_in_place canonicalizes later). load_games_shard() returns the same
(dict, n_games) pair cloud_sweep.load_shard() yields from a materialized parquet.

Replay is vectorized: games advance in lockstep batches, one ply at a time, with
numpy masks over per-game board-state arrays. python-chess is only imported for
the polyglot zobrist table when repetition counts are requested (parity/verify
mode); the training path needs numpy + pyarrow alone.

CLI:
  uv run python lab/decode_games.py --bench games.parquet [--games N]
  uv run python lab/decode_games.py --verify games.parquet golden.parquet
  uv run python lab/decode_games.py --verify-sample games.parquet golden.parquet
  uv run python lab/decode_games.py --aggregates games.parquet golden.parquet
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

HISTORY = 8
BATCH_GAMES = 8192
LICHESS_PREFIX = "https://lichess.org/"
PIECE_VALUES = np.array([0, 1, 3, 3, 5, 9, 0, -1, -3, -3, -5, -9, 0], dtype=np.int32)
PROMO_PIECE = np.array([0, 2, 3, 4, 5], dtype=np.int64)  # move promo code -> white piece code
INIT_BOARD = np.array(
    [4, 2, 3, 5, 6, 3, 2, 4] + [1] * 8 + [0] * 32 + [7] * 8 + [10, 8, 9, 11, 12, 9, 8, 10],
    dtype=np.uint8,
)


def _promotion_table():
    """(from, to, promo-code) -> offset into the sorted 176-class promotion vocab."""
    moves = []
    for source_rank, target_rank in (("7", "8"), ("2", "1")):
        for i in range(8):
            for j in range(max(0, i - 1), min(7, i + 1) + 1):
                for piece in "bnqr":
                    moves.append(f"{chr(97 + i)}{source_rank}{chr(97 + j)}{target_rank}{piece}")
    table = np.full((64, 64, 5), -1, dtype=np.int64)
    letter_code = {"n": 1, "b": 2, "r": 3, "q": 4}
    for offset, uci in enumerate(sorted(moves)):
        frm = (ord(uci[0]) - 97) + 8 * (int(uci[1]) - 1)
        to = (ord(uci[2]) - 97) + 8 * (int(uci[3]) - 1)
        table[frm, to, letter_code[uci[4]]] = offset
    return table


PROMO_TABLE = _promotion_table()

# Castling-rights bitmask: bit0 white kingside, bit1 white queenside, bit2/3 black.
# A move touching a listed square (from or to) clears the same rights python-chess would.
CLEAR_RIGHTS = np.full(64, 0b1111, dtype=np.uint8)
CLEAR_RIGHTS[7] &= 0b1110   # h1 rook
CLEAR_RIGHTS[0] &= 0b1101   # a1 rook
CLEAR_RIGHTS[4] &= 0b1100   # e1 king
CLEAR_RIGHTS[63] &= 0b1011  # h8 rook
CLEAR_RIGHTS[56] &= 0b0111  # a8 rook
CLEAR_RIGHTS[60] &= 0b0011  # e8 king


def _step_tables():
    """Attack geometry with sentinel square 64 for off-board (reads a 255 pad column)."""
    knight = np.full((64, 8), 64, dtype=np.int64)
    king = np.full((64, 8), 64, dtype=np.int64)
    rays = np.full((64, 8, 7), 64, dtype=np.int64)  # dirs 0-3 orthogonal, 4-7 diagonal
    pawn_w = np.full((64, 2), 64, dtype=np.int64)  # black-pawn squares attacking k (white mover)
    pawn_b = np.full((64, 2), 64, dtype=np.int64)  # white-pawn squares attacking k (black mover)
    knight_steps = ((1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1))
    dirs = ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1))
    for sq in range(64):
        r, f = divmod(sq, 8)
        for n, (dr, df) in enumerate(knight_steps):
            if 0 <= r + dr < 8 and 0 <= f + df < 8:
                knight[sq, n] = (r + dr) * 8 + f + df
        for n, (dr, df) in enumerate(dirs):
            if 0 <= r + dr < 8 and 0 <= f + df < 8:
                king[sq, n] = (r + dr) * 8 + f + df
            for step in range(1, 8):
                rr, ff = r + dr * step, f + df * step
                if not (0 <= rr < 8 and 0 <= ff < 8):
                    break
                rays[sq, n, step - 1] = rr * 8 + ff
        for slot, df in enumerate((-1, 1)):  # enemy pawns sit one rank toward the enemy side
            if 0 <= f + df < 8 and r + 1 < 8:
                pawn_w[sq, slot] = (r + 1) * 8 + f + df
            if 0 <= f + df < 8 and r - 1 >= 0:
                pawn_b[sq, slot] = (r - 1) * 8 + f + df
    return knight, king, rays, pawn_w, pawn_b


KNIGHT_TBL, KING_TBL, RAY_TBL, PAWN_SRC_W, PAWN_SRC_B = _step_tables()

_ZOBRIST = None


def _zobrist():
    """Polyglot keys (lazy: python-chess only needed when repetition is requested)."""
    global _ZOBRIST
    if _ZOBRIST is None:
        from chess.polyglot import POLYGLOT_RANDOM_ARRAY as R

        keys = np.zeros((13, 64), dtype=np.uint64)
        for code in range(1, 13):
            white = code <= 6
            ptype = code if white else code - 6
            kind = (ptype - 1) * 2 + (1 if white else 0)
            for sq in range(64):
                keys[code, sq] = R[64 * kind + sq]
        cast = np.zeros(16, dtype=np.uint64)
        for mask in range(16):
            h = 0
            for bit in range(4):
                if mask >> bit & 1:
                    h ^= R[768 + bit]
            cast[mask] = h
        ep_file = np.array([R[772 + f] for f in range(8)], dtype=np.uint64)
        turn = np.uint64(R[780])
        init_pieces = np.bitwise_xor.reduce(keys[INIT_BOARD, np.arange(64)])
        _ZOBRIST = (keys, cast, ep_file, turn, init_pieces)
    return _ZOBRIST


def _ep_capture_safe(board, games, frm, ep_sq, cap_sq, white):
    """Would this en passant capture leave the mover's king un-attacked?

    Simulate the double removal (capturer leaves frm, victim leaves cap_sq, pawn
    lands on ep_sq) and run a full attack scan on the king square: this covers
    pins and the rank-5 double-removal skewer exactly like python-chess legality.
    """
    S = len(games)
    b = board[games].copy()
    rows = np.arange(S)
    b[rows, frm] = 0
    b[rows, cap_sq] = 0
    b[rows, ep_sq] = 1 if white else 7
    bp = np.concatenate([b, np.full((S, 1), 255, dtype=np.uint8)], axis=1)
    k = np.argmax(b == (6 if white else 12), axis=1)
    enemy = 6 if white else 0
    en_p, en_n, en_b, en_r, en_q, en_k = (np.uint8(enemy + pt) for pt in range(1, 7))
    attacked = (bp[rows[:, None], KNIGHT_TBL[k]] == en_n).any(axis=1)
    attacked |= (bp[rows[:, None], KING_TBL[k]] == en_k).any(axis=1)
    pawn_src = PAWN_SRC_W[k] if white else PAWN_SRC_B[k]
    attacked |= (bp[rows[:, None], pawn_src] == en_p).any(axis=1)
    vals = bp[rows[:, None, None], RAY_TBL[k]]  # (S, 8, 7): first hit along each ray
    nz = vals != 0
    first = np.take_along_axis(vals, np.argmax(nz, axis=2)[:, :, None], axis=2)[:, :, 0]
    first = np.where(nz.any(axis=2), first, 0)
    attacked |= ((first[:, :4] == en_r) | (first[:, :4] == en_q)).any(axis=1)
    attacked |= ((first[:, 4:] == en_b) | (first[:, 4:] == en_q)).any(axis=1)
    return ~attacked


def _ep_terms(board, act, ep, white, want_hash, ep_file_keys):
    """Per active game: state[5] (ep square if a LEGAL ep capture exists, else 64)
    and the polyglot ep hash component (present iff a pseudo capturer exists)."""
    A = len(act)
    ep_state = np.full(A, 64, dtype=np.uint8)
    ep_hash = np.zeros(A, dtype=np.uint64) if want_hash else None
    cand = np.nonzero(ep[act] < 64)[0]
    if not len(cand):
        return ep_state, ep_hash
    sub = act[cand]
    e = ep[sub]
    file = e & 7
    base = e - 8 if white else e + 8  # capturer rank; the victim pawn sits on base's file
    pawn_code = 1 if white else 7
    left, right = base - 1, base + 1
    has_left = (file > 0) & (board[sub, np.maximum(left, 0)] == pawn_code)
    has_right = (file < 7) & (board[sub, np.minimum(right, 63)] == pawn_code)
    if want_hash:
        pseudo = has_left | has_right
        ep_hash[cand[pseudo]] = ep_file_keys[file[pseudo]]
    legal = np.zeros(len(cand), dtype=bool)
    for capturer, present in ((left, has_left), (right, has_right)):
        sel = np.nonzero(present & ~legal)[0]
        if len(sel):
            legal[sel] |= _ep_capture_safe(board, sub[sel], capturer[sel], e[sel], base[sel], white)
    ep_state[cand[legal]] = e[legal]
    return ep_state, ep_hash


def decode_batch(mat, counts, want_repetition=False):
    """Replay a lockstep batch of games; rows come back in file order (game-major).

    mat: (B, P) uint16 padded move words; counts: (B,) plies per game.
    """
    B, P = mat.shape
    counts = counts.astype(np.int64)
    n = int(counts.sum())
    row_start = np.zeros(B, dtype=np.int64)
    np.cumsum(counts[:-1], out=row_start[1:])
    squares = np.empty((n, 64), dtype=np.uint8)
    state = np.empty((n, 7), dtype=np.uint8)

    board = np.tile(INIT_BOARD, (B, 1))
    rights = np.full(B, 15, dtype=np.uint8)
    ep = np.full(B, 64, dtype=np.int64)
    halfmove = np.zeros(B, dtype=np.int32)
    balance = np.zeros(B, dtype=np.int32)
    balances = np.zeros((B, P + 1), dtype=np.int16)
    keys = cast = ep_file_keys = turn_key = None
    if want_repetition:
        keys, cast, ep_file_keys, turn_key, init_pieces = _zobrist()
        piece_hash = np.full(B, init_pieces, dtype=np.uint64)
        hashes = np.empty((B, P), dtype=np.uint64)

    for t in range(P):
        act = np.nonzero(counts > t)[0]
        white = (t % 2) == 0
        mv = mat[act, t].astype(np.int64)
        frm = mv & 63
        to = (mv >> 6) & 63
        promo = (mv >> 12) & 7
        rows = row_start[act] + t

        # --- emit the position BEFORE the move ---
        squares[rows] = board[act]
        ep_state, ep_hash = _ep_terms(board, act, ep, white, want_repetition, ep_file_keys)
        r = rights[act]
        state[rows, 0] = 0 if white else 1
        state[rows, 1] = r & 1
        state[rows, 2] = (r >> 1) & 1
        state[rows, 3] = (r >> 2) & 1
        state[rows, 4] = (r >> 3) & 1
        state[rows, 5] = ep_state
        state[rows, 6] = np.minimum(halfmove[act], 100).astype(np.uint8)
        if want_repetition:
            h = piece_hash[act] ^ cast[rights[act]] ^ ep_hash
            if white:
                h = h ^ turn_key
            hashes[act, t] = h

        # --- apply the move ---
        pawn_code = 1 if white else 7
        piece = board[act, frm]
        cap_at_to = board[act, to]
        is_pawn = piece == pawn_code
        ep_capture = is_pawn & ((frm & 7) != (to & 7)) & (cap_at_to == 0)
        cap_sq = np.where(ep_capture, (frm & ~np.int64(7)) | (to & 7), to)
        cap_piece = board[act, cap_sq]
        new_piece = np.where(
            promo > 0, PROMO_PIECE[promo] + (0 if white else 6), piece
        ).astype(np.uint8)
        castle = (piece == (6 if white else 12)) & (np.abs((frm & 7) - (to & 7)) == 2)

        board[act, cap_sq] = 0
        board[act, frm] = 0
        board[act, to] = new_piece
        if castle.any():
            kingside = (to & 7) == 6
            rook_from = np.where(kingside, frm + 3, frm - 4)[castle]
            rook_to = np.where(kingside, frm + 1, frm - 1)[castle]
            cg = act[castle]
            rook_code = 4 if white else 10
            board[cg, rook_from] = 0
            board[cg, rook_to] = rook_code
            if want_repetition:
                piece_hash[cg] ^= keys[rook_code, rook_from] ^ keys[rook_code, rook_to]
        if want_repetition:
            piece_hash[act] ^= keys[piece, frm] ^ keys[new_piece, to] ^ keys[cap_piece, cap_sq]
        rights[act] &= CLEAR_RIGHTS[frm] & CLEAR_RIGHTS[to]
        halfmove[act] = np.where(is_pawn | (cap_piece != 0), 0, halfmove[act] + 1)
        double = is_pawn & (np.abs((frm >> 3) - (to >> 3)) == 2)
        ep[act] = np.where(double, (frm + to) >> 1, 64)
        balance[act] += PIECE_VALUES[new_piece] - PIECE_VALUES[piece] - PIECE_VALUES[cap_piece]
        balances[act, t + 1] = balance[act]

    # --- fully vectorized post-pass over all rows ---
    row_game = np.repeat(np.arange(B), counts)
    row_ply = np.arange(n, dtype=np.int64) - np.repeat(row_start, counts)
    valid = np.arange(P)[None, :] < counts[:, None]
    pad_from = np.full((B, P + HISTORY), 64, dtype=np.uint8)
    pad_to = np.full((B, P + HISTORY), 64, dtype=np.uint8)
    pad_from[:, HISTORY:][valid] = (mat & 63).astype(np.uint8)[valid]
    pad_to[:, HISTORY:][valid] = ((mat >> 6) & 63).astype(np.uint8)[valid]
    window = row_ply[:, None] + np.arange(HISTORY)[None, :]
    game_col = row_game[:, None]
    mv_r = mat[row_game, row_ply].astype(np.int64)
    frm_r, to_r, promo_r = mv_r & 63, (mv_r >> 6) & 63, (mv_r >> 12) & 7
    promo_offset = PROMO_TABLE[frm_r, to_r, promo_r]
    if (promo_offset[promo_r > 0] < 0).any():
        raise ValueError("move word encodes a geometrically impossible promotion")
    target = np.where(promo_r == 0, frm_r * 64 + to_r, 4096 + promo_offset).astype(np.uint16)
    out = {
        "ply": row_ply,
        "squares": squares,
        "state": state,
        "target": target,
        "history_from": pad_from[game_col, window],
        "history_to": pad_to[game_col, window],
        "plies_remaining": counts[row_game] - row_ply,
        "future_material": np.clip(
            balances[row_game, np.minimum(row_ply + 6, counts[row_game])], -127, 127
        ).astype(np.int8),
        "row_game": row_game,
    }
    if want_repetition:
        hrow = hashes[row_game, row_ply]
        order = np.lexsort((hrow, row_game))  # stable: within (game, hash) runs, ply order
        g_s, h_s = row_game[order], hrow[order]
        new_run = np.ones(n, dtype=bool)
        new_run[1:] = (g_s[1:] != g_s[:-1]) | (h_s[1:] != h_s[:-1])
        run_starts = np.nonzero(new_run)[0]
        occurrence = np.arange(n) - run_starts[np.cumsum(new_run) - 1]
        repetition = np.empty(n, dtype=np.uint8)
        repetition[order] = np.minimum(occurrence, 3).astype(np.uint8)
        out["repetition"] = repetition
    return out


def _moves_matrix(batch):
    """Padded (B, P) move matrix + per-game counts from one arrow record batch."""
    col = batch.column("moves")
    counts = pc.list_value_length(col).to_numpy().astype(np.int64)
    declared = batch.column("ply_count").to_numpy().astype(np.int64)
    if not np.array_equal(counts, declared):
        raise ValueError("ply_count column disagrees with moves list lengths")
    if len(counts) and counts.min() <= 0:
        raise ValueError("game with no moves (the reference pipeline drops these)")
    flat = col.flatten().to_numpy().astype(np.uint16)
    B, P = len(counts), int(counts.max()) if len(counts) else 0
    mat = np.zeros((B, P), dtype=np.uint16)
    mat[np.arange(P)[None, :] < counts[:, None]] = flat
    return mat, counts


def load_games_shard(path, offset, batch_games=BATCH_GAMES):
    """Drop-in for cloud_sweep.load_shard() on a compact games parquet.

    Returns ({game_ordinal, squares, state, target, result, history_from,
    history_to, ply, plies_remaining, future_material}, n_games) with the same
    dtypes, values, and row order as loading a lab/prepare.py materialization.
    """
    parquet = pq.ParquetFile(path)
    counts_all = pq.read_table(path, columns=["ply_count"]).column("ply_count").to_numpy()
    n = int(counts_all.astype(np.int64).sum())
    data = {
        "game_ordinal": np.empty(n, dtype=np.int64),
        "squares": np.empty((n, 64), dtype=np.uint8),
        "state": np.empty((n, 7), dtype=np.uint8),
        "target": np.empty(n, dtype=np.int16),
        "result": np.empty(n, dtype=np.uint8),
        "history_from": np.empty((n, HISTORY), dtype=np.uint8),
        "history_to": np.empty((n, HISTORY), dtype=np.uint8),
        "ply": np.empty(n, dtype=np.int16),
        "plies_remaining": np.empty(n, dtype=np.int16),
        "future_material": np.empty(n, dtype=np.int16),
    }
    row = games = 0
    for batch in parquet.iter_batches(
        batch_size=batch_games, columns=["moves", "ply_count", "result"]
    ):
        mat, counts = _moves_matrix(batch)
        part = decode_batch(mat, counts)
        nb = len(part["ply"])
        sl = slice(row, row + nb)
        for name in ("squares", "state", "target", "history_from", "history_to",
                     "ply", "plies_remaining", "future_material"):
            data[name][sl] = part[name]
        game_result = batch.column("result").to_numpy()
        data["result"][sl] = np.repeat(game_result.astype(np.uint8), counts)
        data["game_ordinal"][sl] = np.repeat(
            offset + games + np.arange(len(counts), dtype=np.int64), counts
        )
        row += nb
        games += len(counts)
    assert row == n
    return data, games


GAME_COLUMNS = ["game_id", "white_elo", "black_elo", "result", "ply_count", "moves"]


def _full_rows(batch):
    """Decode one record batch of games into fully enriched rows (all prepare.py
    columns incl. repetition, elos, and the reconstructed lichess-URL game_id)."""
    mat, counts = _moves_matrix(batch)
    part = decode_batch(mat, counts, want_repetition=True)
    ids = np.array(
        [LICHESS_PREFIX + gid.decode() for gid in batch.column("game_id").to_pylist()],
        dtype=object,
    )
    part["game_id"] = np.repeat(ids, counts)
    part["white_elo"] = np.repeat(batch.column("white_elo").to_numpy(), counts)
    part["black_elo"] = np.repeat(batch.column("black_elo").to_numpy(), counts)
    part["result"] = np.repeat(batch.column("result").to_numpy().astype(np.uint8), counts)
    return part


def iter_full_batches(path, batch_games=BATCH_GAMES):
    """Yield fully enriched row chunks in exact file row order."""
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_games, columns=GAME_COLUMNS):
        yield _full_rows(batch)


VERIFY_COLUMNS = [
    "game_id", "ply", "squares", "state", "target", "result", "history_from",
    "history_to", "repetition", "plies_remaining", "future_material",
    "white_elo", "black_elo",
]


def _golden_numpy(batch):
    out = {}
    for name in VERIFY_COLUMNS:
        col = batch.column(name)
        if isinstance(col, pa.ChunkedArray):
            col = col.combine_chunks()
        if pa.types.is_fixed_size_list(col.type):
            width = col.type.list_size
            out[name] = col.flatten().to_numpy().reshape(-1, width)
        elif name == "game_id":
            out[name] = np.array(col.to_pylist(), dtype=object)
        else:
            out[name] = col.to_numpy()
    return out


def _compare_chunks(golden, decoded, row_base, mismatches, mismatch_limit):
    """Column-by-column bit compare of two aligned row chunks; appends mismatches."""
    per_column = {}
    for name in VERIFY_COLUMNS:
        a, b = golden[name], decoded[name]
        equal = np.array_equal(a, b)
        per_column[name] = equal
        if not equal and len(mismatches) < mismatch_limit:
            diff = np.nonzero((a != b).any(axis=1) if a.ndim > 1 else (a != b))[0]
            for d in diff[: mismatch_limit - len(mismatches)]:
                mismatches.append({
                    "row": int(row_base + d),
                    "column": name,
                    "golden": np.asarray(a[d]).tolist(),
                    "decoded": np.asarray(b[d]).tolist(),
                    "game_id": str(golden["game_id"][d]),
                })
    return per_column


def verify(games_path, golden_path, mismatch_limit=10):
    """Exhaustive streaming row/column comparison of decoded rows vs the golden
    materialized parquet. Returns a report dict; mismatches abort after the limit."""
    decoded = iter_full_batches(games_path)
    buffer, buffer_pos = None, 0
    rows_done = 0
    mismatches = []
    started = time.perf_counter()
    golden_file = pq.ParquetFile(golden_path)
    exhausted_early = False
    for batch in golden_file.iter_batches(batch_size=65536):
        golden = _golden_numpy(batch)
        gn, gpos = len(golden["ply"]), 0
        while gpos < gn:
            if buffer is None or buffer_pos >= len(buffer["ply"]):
                buffer = next(decoded, None)
                buffer_pos = 0
                if buffer is None:
                    exhausted_early = True
                    break
            take = min(gn - gpos, len(buffer["ply"]) - buffer_pos)
            for name in VERIFY_COLUMNS:
                a = golden[name][gpos : gpos + take]
                b = buffer[name][buffer_pos : buffer_pos + take]
                if not np.array_equal(a, b):
                    diff = np.nonzero(
                        (a != b).any(axis=1) if a.ndim > 1 else (a != b)
                    )[0]
                    for d in diff[: max(1, mismatch_limit - len(mismatches))]:
                        mismatches.append({
                            "row": int(rows_done + gpos + d),
                            "column": name,
                            "golden": np.asarray(a[d]).tolist(),
                            "decoded": np.asarray(b[d]).tolist(),
                            "game_id": str(golden["game_id"][gpos + d]),
                        })
                    if len(mismatches) >= mismatch_limit:
                        return _verify_report(rows_done, golden_file, mismatches,
                                              started, aborted=True)
            gpos += take
            buffer_pos += take
        rows_done += gn
        if exhausted_early:
            break
        if rows_done % (65536 * 64) < 65536:
            print(json.dumps({"verified_rows": rows_done,
                              "elapsed": round(time.perf_counter() - started, 1)}), flush=True)
    decoded_leftover = 0
    if buffer is not None:
        decoded_leftover = len(buffer["ply"]) - buffer_pos
    for extra in decoded:
        decoded_leftover += len(extra["ply"])
    return _verify_report(rows_done, golden_file, mismatches, started,
                          aborted=False, exhausted_early=exhausted_early,
                          decoded_leftover=decoded_leftover)


def _verify_report(rows_done, golden_file, mismatches, started,
                   aborted=False, exhausted_early=False, decoded_leftover=0):
    report = {
        "golden_rows": golden_file.metadata.num_rows,
        "rows_compared": rows_done,
        "mismatches": mismatches,
        "aborted_at_limit": aborted,
        "decoded_ran_out_early": exhausted_early,
        "decoded_rows_beyond_golden": decoded_leftover,
        "identical": (not mismatches and not aborted and not exhausted_early
                      and decoded_leftover == 0
                      and rows_done == golden_file.metadata.num_rows),
        "wall_seconds": round(time.perf_counter() - started, 1),
    }
    return report


def _read_golden_rows(golden_file, group_starts, lo, hi):
    """Read golden rows [lo, hi) by covering row groups, sliced exactly."""
    g0 = int(np.searchsorted(group_starts, lo, side="right") - 1)
    g1 = g0
    while group_starts[g1 + 1] < hi:
        g1 += 1
    table = golden_file.read_row_groups(list(range(g0, g1 + 1)), columns=VERIFY_COLUMNS)
    offset = lo - int(group_starts[g0])
    return _golden_numpy(table.slice(offset, hi - lo).combine_chunks())


def verify_sampled(games_path, golden_path, blocks=48, block_games=150,
                   edge_games=100, seed=20260808, mismatch_limit=10):
    """Game-aligned sampled parity: the first and last `edge_games` games plus
    `blocks` random blocks of `block_games` consecutive games spread uniformly
    across the shard, compared bit-for-bit against the golden parquet on every
    output column. Alignment uses the games file's ply_count cumsum, so any
    ordering or row-count drift shows up as loud game_id/column mismatches."""
    started = time.perf_counter()
    games_table = pq.read_table(games_path, columns=GAME_COLUMNS)
    counts_all = games_table.column("ply_count").to_numpy().astype(np.int64)
    n_games = len(counts_all)
    row_offsets = np.zeros(n_games + 1, dtype=np.int64)
    np.cumsum(counts_all, out=row_offsets[1:])
    golden_file = pq.ParquetFile(golden_path)
    golden_rows = golden_file.metadata.num_rows
    group_starts = np.zeros(golden_file.metadata.num_row_groups + 1, dtype=np.int64)
    for i in range(golden_file.metadata.num_row_groups):
        group_starts[i + 1] = group_starts[i] + golden_file.metadata.row_group(i).num_rows

    spans = [(0, min(edge_games, n_games)),
             (max(0, n_games - edge_games), n_games)]
    rng = np.random.default_rng(seed)
    interior = max(0, n_games - block_games)
    for start in sorted(rng.integers(0, interior + 1, size=blocks).tolist()):
        spans.append((start, min(start + block_games, n_games)))
    # merge overlaps so no row is compared twice and reads stay sequential-ish
    spans.sort()
    merged = [spans[0]]
    for a, b in spans[1:]:
        if a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))

    mismatches = []
    column_ok = {name: True for name in VERIFY_COLUMNS}
    games_compared = rows_compared = 0
    total_row_match = int(row_offsets[-1]) == golden_rows
    for a, b in merged:
        lo, hi = int(row_offsets[a]), int(row_offsets[b])
        if hi > golden_rows:  # keep going: mismatch reporting will localize the drift
            hi = golden_rows
            if lo >= hi:
                continue
        batch = games_table.slice(a, b - a).combine_chunks().to_batches()[0]
        decoded = _full_rows(batch)
        golden = _read_golden_rows(golden_file, group_starts, lo, hi)
        take = min(len(golden["ply"]), len(decoded["ply"]))
        golden = {k: v[:take] for k, v in golden.items()}
        clipped = {k: decoded[k][:take] for k in VERIFY_COLUMNS}
        per_column = _compare_chunks(golden, clipped, lo, mismatches, mismatch_limit)
        for name, ok in per_column.items():
            column_ok[name] &= ok
        games_compared += b - a
        rows_compared += take
    return {
        "mode": "sampled",
        "games_in_shard": n_games,
        "games_compared": games_compared,
        "rows_compared": rows_compared,
        "golden_rows": golden_rows,
        "decoded_total_rows": int(row_offsets[-1]),
        "total_row_count_match": total_row_match,
        "column_pass": column_ok,
        "mismatches": mismatches,
        "identical": total_row_match and not mismatches and all(column_ok.values()),
        "wall_seconds": round(time.perf_counter() - started, 1),
    }


AGG_COLUMNS = [c for c in VERIFY_COLUMNS if c != "game_id"]


def _accumulate(agg, chunk):
    for name in AGG_COLUMNS:
        v = np.asarray(chunk[name])
        entry = agg.setdefault(name, {"sum": 0, "min": None, "max": None, "nonzero": 0, "count": 0})
        entry["sum"] += int(v.astype(np.int64).sum())
        lo, hi = int(v.min()), int(v.max())
        entry["min"] = lo if entry["min"] is None else min(entry["min"], lo)
        entry["max"] = hi if entry["max"] is None else max(entry["max"], hi)
        entry["nonzero"] += int(np.count_nonzero(v))
        entry["count"] += v.size


def aggregates(games_path, golden_path):
    """Full-shard per-column aggregates (sum/min/max/nonzero/count) on the decoded
    stream vs the golden parquet: catches pervasive systematic offsets that a
    sample might undersell, without holding either side in memory."""
    started = time.perf_counter()
    dec_agg = {}
    dec_rows = 0
    for chunk in iter_full_batches(games_path):
        _accumulate(dec_agg, chunk)
        dec_rows += len(chunk["ply"])
    decode_seconds = time.perf_counter() - started
    gold_agg = {}
    gold_rows = 0
    for batch in pq.ParquetFile(golden_path).iter_batches(batch_size=65536,
                                                          columns=VERIFY_COLUMNS):
        chunk = _golden_numpy(batch)
        _accumulate(gold_agg, chunk)
        gold_rows += len(chunk["ply"])
    column_pass = {name: dec_agg.get(name) == gold_agg.get(name) for name in AGG_COLUMNS}
    return {
        "mode": "aggregates",
        "decoded_rows": dec_rows,
        "golden_rows": gold_rows,
        "row_count_match": dec_rows == gold_rows,
        "column_pass": column_pass,
        "decoded": dec_agg,
        "golden": gold_agg,
        "identical": dec_rows == gold_rows and all(column_pass.values()),
        "decode_positions_per_second": round(dec_rows / max(1e-9, decode_seconds)),
        "wall_seconds": round(time.perf_counter() - started, 1),
    }


def bench(path, max_games=0):
    started = time.perf_counter()
    parquet = pq.ParquetFile(path)
    rows = games = 0
    for batch in parquet.iter_batches(batch_size=BATCH_GAMES,
                                      columns=["moves", "ply_count", "result"]):
        mat, counts = _moves_matrix(batch)
        part = decode_batch(mat, counts)
        rows += len(part["ply"])
        games += len(counts)
        if max_games and games >= max_games:
            break
    elapsed = time.perf_counter() - started
    return {"games": games, "positions": rows, "wall_seconds": round(elapsed, 2),
            "positions_per_second": round(rows / elapsed)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", metavar="GAMES_PARQUET")
    parser.add_argument("--games", type=int, default=0, help="bench: stop after N games")
    parser.add_argument("--verify", nargs=2, metavar=("GAMES_PARQUET", "GOLDEN_PARQUET"))
    parser.add_argument("--verify-sample", nargs=2, metavar=("GAMES_PARQUET", "GOLDEN_PARQUET"))
    parser.add_argument("--aggregates", nargs=2, metavar=("GAMES_PARQUET", "GOLDEN_PARQUET"))
    parser.add_argument("--blocks", type=int, default=48)
    parser.add_argument("--block-games", type=int, default=150)
    parser.add_argument("--seed", type=int, default=20260808)
    args = parser.parse_args()
    if args.bench:
        print(json.dumps(bench(args.bench, args.games)))
    elif args.verify:
        report = verify(args.verify[0], args.verify[1])
        print(json.dumps(report, indent=2))
        raise SystemExit(0 if report["identical"] else 1)
    elif args.verify_sample:
        report = verify_sampled(args.verify_sample[0], args.verify_sample[1],
                                blocks=args.blocks, block_games=args.block_games,
                                seed=args.seed)
        print(json.dumps(report, indent=2))
        raise SystemExit(0 if report["identical"] else 1)
    elif args.aggregates:
        report = aggregates(args.aggregates[0], args.aggregates[1])
        print(json.dumps(report, indent=2))
        raise SystemExit(0 if report["identical"] else 1)
    else:
        parser.error("pass --bench, --verify, --verify-sample, or --aggregates")


if __name__ == "__main__":
    main()
