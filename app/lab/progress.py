from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

ProgressKind = Literal["started", "progress", "completed", "message"]


@dataclass(frozen=True)
class ProgressEvent:
    stage: str
    kind: ProgressKind = "progress"
    current: int | float | None = None
    total: int | float | None = None
    message: str = ""
    details: dict[str, object] | None = None

    @property
    def fraction(self) -> float | None:
        if self.current is None or self.total is None or self.total <= 0:
            return None
        return max(0.0, min(1.0, float(self.current) / float(self.total)))


class ProgressCallback(Protocol):
    def __call__(self, event: ProgressEvent) -> None: ...


def emit(callback: ProgressCallback | None, event: ProgressEvent) -> None:
    if callback is not None:
        callback(event)

