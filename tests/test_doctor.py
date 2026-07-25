import torch

from chess_gpt.doctor import autodiff_smoke, environment_report, select_device


def test_autodiff_smoke_on_cpu() -> None:
    result = autodiff_smoke(torch.device("cpu"))

    assert result == {"x": 3.0, "y": 9.0, "dy_dx": 6.0}


def test_selected_device_is_supported() -> None:
    assert select_device().type in {"cpu", "cuda", "mps"}


def test_environment_report_contains_reproducibility_basics() -> None:
    report = environment_report()

    assert {"python", "platform", "torch", "device", "autodiff"} <= report.keys()
