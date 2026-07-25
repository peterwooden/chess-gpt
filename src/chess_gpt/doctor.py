"""Report whether the locked local environment can perform basic autodiff."""

from __future__ import annotations

import json
import platform
import sys
from typing import Any

import torch


def select_device() -> torch.device:
    """Prefer Apple MPS, then CUDA, while retaining a dependable CPU fallback."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def autodiff_smoke(device: torch.device) -> dict[str, float]:
    """Differentiate y=x² at x=3; dy/dx must be 6."""
    x = torch.tensor([3.0], device=device, requires_grad=True)
    y = x.square().sum()
    y.backward()
    if x.grad is None:
        raise RuntimeError("autograd did not populate x.grad")
    return {"x": x.item(), "y": y.item(), "dy_dx": x.grad.item()}


def environment_report() -> dict[str, Any]:
    device = select_device()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": device.type,
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "autodiff": autodiff_smoke(device),
    }


def main() -> None:
    json.dump(environment_report(), sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
