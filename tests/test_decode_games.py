"""lab/decode_games.py must reproduce lab/prepare.py's rows bit for bit.

Fixtures are hand-picked games converted to the compact move-word schema with
python-chess, covering castling on both wings, en passant (captured, declined,
and pseudo-legal-but-illegal via the rank-5 skewer), all four promotion pieces
including underpromotion with capture, mate endings, repetition up to the cap,
halfmove clocks past the 100 clamp, and all three results. The reference
encoding is prepare.rows_from_moves itself — the python-chess-driven pipeline
that produced every existing shard.
"""

import importlib.util
import random
import sys
from pathlib import Path

import chess
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

LAB = Path(__file__).resolve().parents[1] / "lab"


def _load_module(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, LAB / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


decode_games = _load_module("decode_games")
prepare = _load_module("prepare")

# (uci moves, result in the games/prepare convention: 0 white won, 1 draw, 2 black won)
HANDCRAFTED = [
    # kingside castling both sides
    ("e2e4 e7e5 g1f3 b8c6 f1c4 g8f6 e1g1 f8c5 b1c3 e8g8 d2d3 d7d6 c1g5 c8g4", 0),
    # queenside castling both sides
    ("d2d4 d7d5 b1c3 b8c6 c1f4 c8f5 d1d2 d8d7 e1c1 e8c8 g1f3 g8f6 e2e3 e7e6", 1),
    # en passant captured
    ("e2e4 g8f6 e4e5 d7d5 e5d6 c7d6 d2d4 b8c6 g1f3 e7e5", 2),
    # en passant legal but declined (state[5] must show the ep square)
    ("e2e4 c7c5 e4e5 d7d5 g1f3 e7e6 d2d4 b8c6", 1),
    # en passant pseudo-possible but ILLEGAL: rank-5 skewer against the king
    ("g2g4 c7c6 g4g5 d8a5 f2f3 d7d6 e1f2 e7e6 f2g3 b8d7 g3h4 g8e7 h4h5 f7f5 h5h4 a5b6", 1),
    # promotion to R with capture, black underpromotion to B with capture
    ("a2a4 h7h5 a4a5 h5h4 a5a6 h4h3 a6b7 h3g2 b7a8r g2h1b a8b8 h1e4 b1c3 e4g6", 0),
    # straight-push underpromotion to N
    ("a2a4 b7b5 a4b5 b8a6 b5b6 a6c5 b6b7 g7g6 b7b8n f8g7 b8c6 d7c6", 2),
    # promotion to Q with capture
    ("b2b4 a7a5 b4a5 b8c6 a5a6 c6b4 a6b7 b4d5 b7c8q d8c8 b1c3 e7e5", 1),
    # scholar's mate (ends in checkmate, white win)
    ("e2e4 e7e5 d1h5 b8c6 f1c4 g8f6 h5f7", 0),
    # fool's mate (black win)
    ("f2f3 e7e5 g2g4 d8h4", 2),
    # repetition to the min(.,3) cap and halfmove clock past the 100 clamp
    ("g1f3 g8f6 f3g1 f6g8 " * 30 + "e2e4 c7c5", 1),
    # castling rights lost by rook then king moves
    ("h2h4 a7a5 h1h3 a8a6 h3g3 a6b6 e2e4 e7e5 f1c4 f8c5 e1e2 e8e7 d2d3 d7d6", 2),
    # same piece arrangement recurring with vs without a polyglot ep component
    ("e2e4 d7d5 e4e5 d5d4 c2c4 g8f6 g1f3 f6g8 f3g1 b8c6", 1),
    # same arrangement recurring with different castling rights (no false repetition)
    ("h2h4 h7h5 h1h3 h8h6 h3h1 h6h8 h1h3 h8h6 h3h1 h6h8 e2e4 e7e5", 1),
]
PROMO_CODE = {chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4}
GAMES_SCHEMA = pa.schema(
    [
        pa.field("game_id", pa.binary(8)),
        pa.field("white_elo", pa.uint16()),
        pa.field("black_elo", pa.uint16()),
        pa.field("result", pa.int8()),
        pa.field("time_base_s", pa.uint16()),
        pa.field("time_inc_s", pa.uint8()),
        pa.field("termination", pa.int8()),
        pa.field("ply_count", pa.uint16()),
        pa.field("moves", pa.list_(pa.uint16())),
    ]
)


def _random_game(seed, max_plies=90):
    rng = random.Random(seed)
    board = chess.Board()
    moves = []
    while len(moves) < max_plies and not board.is_game_over():
        move = rng.choice(sorted(board.legal_moves, key=lambda m: m.uci()))
        moves.append(move)
        board.push(move)
    result = {"1-0": 0, "1/2-1/2": 1, "0-1": 2}.get(board.result(claim_draw=False), 1)
    return moves, result


def _fixture_games():
    games = []
    for i, (ucis, result) in enumerate(HANDCRAFTED):
        moves = [chess.Move.from_uci(u) for u in ucis.split()]
        games.append((f"hc{i:06d}", moves, result, 1500 + i, 1601 + i))
    for j in range(6):
        moves, result = _random_game(1000 + j)
        games.append((f"rp{j:06d}", moves, result, 2000 + j, 2101 + j))
    assert len(games) == 20
    return games


def _move_words(moves):
    board = chess.Board()
    words = []
    for move in moves:
        assert move in board.legal_moves, f"fixture bug: illegal {move.uci()}"
        promo = PROMO_CODE[move.promotion] if move.promotion else 0
        words.append(move.from_square | (move.to_square << 6) | (promo << 12))
        board.push(move)
    return words


@pytest.fixture(scope="module")
def fixture_games():
    return _fixture_games()


@pytest.fixture(scope="module")
def games_parquet(fixture_games, tmp_path_factory):
    path = tmp_path_factory.mktemp("games") / "games.parquet"
    table = pa.Table.from_pylist(
        [
            {
                "game_id": gid.encode(),
                "white_elo": white_elo,
                "black_elo": black_elo,
                "result": result,
                "time_base_s": 300,
                "time_inc_s": 3,
                "termination": 0,
                "ply_count": len(moves),
                "moves": _move_words(moves),
            }
            for gid, moves, result, white_elo, black_elo in fixture_games
        ],
        schema=GAMES_SCHEMA,
    )
    pq.write_table(table, path)
    return path


@pytest.fixture(scope="module")
def reference_rows(fixture_games):
    """The old pipeline's encoding, game by game (prepare.rows_from_moves)."""
    per_game = []
    for gid, moves, result, white_elo, black_elo in fixture_games:
        rows = prepare.rows_from_moves(
            moves, decode_games.LICHESS_PREFIX + gid, result,
            keep_color=None, global_counts=None, dedup_cap=0,
            elos=(white_elo, black_elo),
        )
        assert rows, f"fixture bug: no rows for {gid}"
        per_game.append(rows)
    return per_game


@pytest.fixture(scope="module")
def golden_parquet(reference_rows, tmp_path_factory):
    path = tmp_path_factory.mktemp("golden") / "golden.parquet"
    flat = [row for rows in reference_rows for row in rows]
    pq.write_table(pa.Table.from_pylist(flat, schema=prepare.SCHEMA), path)
    return path


def _decoded_columns(games_parquet, batch_games):
    chunks = list(decode_games.iter_full_batches(games_parquet, batch_games=batch_games))
    return {
        name: np.concatenate([chunk[name] for chunk in chunks])
        for name in decode_games.VERIFY_COLUMNS
    }


@pytest.mark.parametrize("batch_games", [64, 7])  # one batch, and ragged multi-batch
def test_decoded_rows_match_reference(games_parquet, reference_rows, fixture_games, batch_games):
    decoded = _decoded_columns(games_parquet, batch_games)
    flat = [row for rows in reference_rows for row in rows]
    assert len(decoded["ply"]) == len(flat)
    row = 0
    for (gid, _, _, _, _), rows in zip(fixture_games, reference_rows, strict=True):
        for ply, ref in enumerate(rows):
            for name in decode_games.VERIFY_COLUMNS:
                got = decoded[name][row]
                want = ref[name]
                got = got.tolist() if isinstance(got, np.ndarray) else got
                want = list(want) if isinstance(want, (list, tuple)) else want
                assert got == want, (
                    f"game {gid} ply {ply} column {name}: decoded {got!r} != reference {want!r}"
                )
            row += 1


def test_fixture_exercises_the_contract(reference_rows):
    """Guard the fixture itself: the tricky encodings must actually appear."""
    flat = [row for rows in reference_rows for row in rows]
    targets = {row["target"] for row in flat}
    assert any(t >= 4096 for t in targets), "no promotion targets"
    # all four promotion pieces present in the raw fixtures
    ucis = " ".join(u for u, _ in HANDCRAFTED)
    assert all(any(u.endswith(p) for u in ucis.split()) for p in "qrbn")
    assert any(row["state"][5] != 64 for row in flat), "no legal-ep state emitted"
    assert any(row["repetition"] == 3 for row in flat), "repetition cap never reached"
    assert any(row["state"][6] == 100 for row in flat), "halfmove clamp never reached"
    results = {row["result"] for row in flat}
    assert results == {0, 1, 2}


def test_ep_skewer_position_reads_no_ep(reference_rows, fixture_games):
    """The rank-5 skewer game: after ...f7f5 the ep capture is pseudo-legal but
    illegal, so state[5] must be 64 — and the reference agrees."""
    index = 4
    rows = reference_rows[index]
    row = rows[14]  # white to move right after f7f5
    assert row["state"][5] == 64


def test_load_games_shard_matches_load_shard(games_parquet, golden_parquet):
    cloud_sweep = _load_module("cloud_sweep")
    want, want_games = cloud_sweep.load_shard(str(golden_parquet), offset=7)
    got, got_games = decode_games.load_games_shard(str(games_parquet), offset=7)
    assert got_games == want_games == 20
    assert set(got) == set(want)
    for name in want:
        assert got[name].dtype == want[name].dtype, name
        assert np.array_equal(got[name], want[name]), name


def test_verify_sampled_reports_identical(games_parquet, golden_parquet):
    report = decode_games.verify_sampled(
        str(games_parquet), str(golden_parquet), blocks=3, block_games=5, edge_games=4
    )
    assert report["identical"], report
    assert report["total_row_count_match"]


def test_verify_and_aggregates_catch_corruption(games_parquet, golden_parquet, tmp_path):
    table = pq.read_table(games_parquet)
    moves = table.column("moves").to_pylist()
    moves[9] = [*moves[9][:-1], moves[9][-1] ^ 1]  # nudge one to-square
    corrupt = table.set_column(
        table.schema.get_field_index("moves"), "moves",
        pa.array(moves, type=pa.list_(pa.uint16())),
    )
    path = tmp_path / "corrupt.parquet"
    pq.write_table(corrupt, path)
    sampled = decode_games.verify_sampled(
        str(path), str(golden_parquet), blocks=3, block_games=20, edge_games=20
    )
    assert not sampled["identical"]
    assert sampled["mismatches"]
    agg = decode_games.aggregates(str(path), str(golden_parquet))
    assert not agg["identical"]


def test_full_verify_streaming(games_parquet, golden_parquet):
    report = decode_games.verify(str(games_parquet), str(golden_parquet))
    assert report["identical"], report
