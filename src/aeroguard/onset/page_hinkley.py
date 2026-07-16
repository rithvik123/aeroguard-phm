"""Small deterministic Page-Hinkley change detector."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PageHinkley:
    """One-sided Page-Hinkley detector for mean shifts."""

    delta: float = 0.005
    threshold: float = 5.0
    min_observations: int = 20
    direction: str = "increase"
    reset_after_detection: bool = False

    def __post_init__(self) -> None:
        if self.delta < 0:
            raise ValueError("delta must be non-negative.")
        if self.threshold <= 0:
            raise ValueError("threshold must be positive.")
        if self.min_observations <= 0:
            raise ValueError("min_observations must be positive.")
        if self.direction not in {"increase", "decrease"}:
            raise ValueError("direction must be 'increase' or 'decrease'.")
        self.reset()

    def reset(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.cumulative = 0.0
        self.minimum_cumulative = 0.0
        self.detected_once = False

    def update(self, value: float) -> bool:
        signal = -float(value) if self.direction == "decrease" else float(value)
        self.count += 1
        self.mean += (signal - self.mean) / self.count
        self.cumulative += signal - self.mean - self.delta
        self.minimum_cumulative = min(self.minimum_cumulative, self.cumulative)
        change = (
            self.count >= self.min_observations
            and (self.cumulative - self.minimum_cumulative) > self.threshold
        )
        if change:
            if self.reset_after_detection:
                self.reset()
            else:
                self.detected_once = True
        return bool(change)

    def run(self, values: list[float]) -> list[bool]:
        return [self.update(value) for value in values]
