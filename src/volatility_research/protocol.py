"""Load and verify the frozen protocol before evaluating any prices."""

import hashlib
import tomllib
from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass(frozen=True)
class Protocol:
    sha256: str
    evaluation_start: date
    evaluation_end: date
    historical_window: int
    ewma_decay: float
    garch_window: int
    variance_floor: float
    maximum_garch_fallback_fraction: float
    minimum_evaluation_count: int


def load_protocol(path: Path) -> Protocol:
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    expected = path.with_suffix(".sha256").read_text().strip()
    if digest != expected:
        raise ValueError(
            "Protocol hash mismatch: preserve the original and use a new study identity"
        )
    values = tomllib.loads(payload.decode())
    if values["study_id"] != "spy-daily-variance-v1":
        raise ValueError("Unsupported study identity")
    return Protocol(
        sha256=digest,
        evaluation_start=date.fromisoformat(values["evaluation_start"]),
        evaluation_end=date.fromisoformat(values["evaluation_end"]),
        historical_window=values["historical_window"],
        ewma_decay=values["ewma_decay"],
        garch_window=values["garch_window"],
        variance_floor=values["variance_floor"],
        maximum_garch_fallback_fraction=values["maximum_garch_fallback_fraction"],
        minimum_evaluation_count=values["minimum_evaluation_count"],
    )
