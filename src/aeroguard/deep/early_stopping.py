"""Early stopping state."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EarlyStopping:
    patience: int
    min_delta: float = 0.0

    def __post_init__(self) -> None:
        if self.patience < 0:
            raise ValueError("patience must be non-negative.")
        self.best_value: float | None = None
        self.best_epoch: int = 0
        self.bad_epochs: int = 0

    def update(self, value: float, epoch: int) -> bool:
        if self.best_value is None or value < self.best_value - self.min_delta:
            self.best_value = float(value)
            self.best_epoch = int(epoch)
            self.bad_epochs = 0
            return True
        self.bad_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs > self.patience

