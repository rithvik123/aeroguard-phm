"""Reproducibility helpers for PyTorch training."""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        import torch

        torch.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.benchmark = False
        else:
            torch.use_deterministic_algorithms(False)
            torch.backends.cudnn.benchmark = True
    except Exception:
        pass
    os.environ["PYTHONHASHSEED"] = str(int(seed))

