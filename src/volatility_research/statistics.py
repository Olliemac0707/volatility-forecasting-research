"""Forecast scores and a predeclared serial-dependence-aware uncertainty estimate."""

import math
from dataclasses import asdict, dataclass

import numpy as np

from volatility_research.models import Array


def qlike(target: Array, forecast: Array) -> Array:
    if target.shape != forecast.shape or target.ndim != 1:
        raise ValueError("Targets and forecasts must have the same vector shape")
    if not np.isfinite(target).all() or np.any(target < 0):
        raise ValueError("Invalid squared-return target")
    if not np.isfinite(forecast).all() or np.any(forecast <= 0):
        raise ValueError("Forecast variances must be finite and positive")
    return np.log(forecast) + target / forecast


@dataclass(frozen=True)
class HacInterval:
    count: int
    lag: int
    mean_difference: float
    standard_error: float
    lower_95: float
    upper_95: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def hac_interval(differences: Array) -> HacInterval:
    n = len(differences)
    if differences.ndim != 1 or n < 30 or not np.isfinite(differences).all():
        raise ValueError("HAC requires at least 30 finite paired loss differences")
    lag = min(n - 1, math.floor(4 * (n / 100) ** (2 / 9)))
    average = float(np.mean(differences))
    centered = differences - average
    long_run = float(centered @ centered) / n
    for j in range(1, lag + 1):
        covariance = float(centered[j:] @ centered[:-j]) / n
        long_run += 2 * (1 - j / (lag + 1)) * covariance
    if long_run < -1e-12:
        raise ValueError("Invalid negative HAC long-run variance")
    se = math.sqrt(max(0.0, long_run) / n)
    z = 1.959963984540054
    return HacInterval(n, lag, average, se, average - z * se, average + z * se)
