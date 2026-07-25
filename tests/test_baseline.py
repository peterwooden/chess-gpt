from __future__ import annotations

from pathlib import Path

import chess

from chess_gpt.baseline import SanNgramModel, decode_token_stream, self_play


def trained_toy_model() -> SanNgramModel:
    model = SanNgramModel(order=2)
    model.fit(
        [
            ["e4", "e5", "Nf3", "Nc6", "Bb5"],
            ["e4", "e5", "Nf3", "Nc6", "Bc4"],
            ["d4", "d5", "c4"],
        ]
    )
    model.prune()
    return model


def test_factorized_tokens_reassemble_as_san() -> None:
    assert decode_token_stream("[??] [+] [x] [=Q] [ef8]") == ["exf8=Q+"]
    assert decode_token_stream("[Nf3] [x] [Bb5] [?] [e5]") == ["Nf3", "Bxb5", "e5"]


def test_model_predicts_a_legal_learned_continuation() -> None:
    model = trained_toy_model()

    assert model.predict([]).san == "e4"
    assert model.predict(["e4"]).san == "e5"
    assert model.predict(["e4", "e5"]).san == "Nf3"


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.json.gz"
    trained_toy_model().save(checkpoint, metadata={"purpose": "test"})

    loaded = SanNgramModel.load(checkpoint)

    assert loaded.predict(["e4", "e5", "Nf3"]).san == "Nc6"


def test_self_play_only_adds_legal_moves() -> None:
    game = self_play(trained_toy_model(), max_plies=16)
    board = chess.Board()

    for move in game.mainline_moves():
        assert move in board.legal_moves
        board.push(move)

    assert board.ply() == 16 or board.is_game_over(claim_draw=True)
