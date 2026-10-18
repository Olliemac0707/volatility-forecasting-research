"""Independent score calculations, dependence checks, data quality and protocol identity."""

import csv
from pathlib import Path

import numpy as np
import pytest

from volatility_research.data import read_prices, validate_calendar
from volatility_research.protocol import load_protocol
from volatility_research.statistics import hac_interval, qlike


def test_qlike_matches_manual_calculation_and_handles_zero_target():
    result = qlike(np.array([0.0, 0.04]), np.array([0.01, 0.02]))
    np.testing.assert_allclose(result, [np.log(0.01), np.log(0.02) + 2])


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_qlike_rejects_invalid_forecast(bad):
    with pytest.raises(ValueError):
        qlike(np.array([0.01]), np.array([bad]))


def test_hac_matches_independent_covariance_sum():
    values = np.sin(np.arange(120) / 6) + 0.2
    interval = hac_interval(values)
    centered = values - values.mean()
    lrv = sum(float(v * v) for v in centered) / len(values)
    for lag in range(1, interval.lag + 1):
        gamma = sum(centered[t] * centered[t - lag] for t in range(lag, len(values))) / len(values)
        lrv += 2 * (1 - lag / (interval.lag + 1)) * gamma
    assert interval.standard_error == pytest.approx(np.sqrt(lrv / len(values)))
    assert interval.lower_95 < interval.mean_difference < interval.upper_95
    assert interval.standard_error > values.std(ddof=0) / np.sqrt(len(values))


def test_zero_paired_difference_is_not_artificial_significance():
    result = hac_interval(np.zeros(100))
    assert result.lower_95 == result.upper_95 == result.mean_difference == 0


@pytest.mark.parametrize(
    "rows",
    [
        [("2020-01-02", "100"), ("2020-01-02", "101")],
        [("2020-01-03", "100"), ("2020-01-02", "101")],
        [("2020-01-02", "nan"), ("2020-01-03", "101")],
        [("2020-01-02", "0"), ("2020-01-03", "101")],
    ],
)
def test_bad_prices_rejected(tmp_path, rows):
    path = tmp_path / "prices.csv"
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["date", "adj_close"])
        writer.writerows(rows)
    with pytest.raises(ValueError):
        read_prices(path)


def test_missing_exchange_sessions_cannot_be_silently_skipped():
    with pytest.raises(ValueError, match="calendar mismatch"):
        validate_calendar([])


def test_protocol_modification_detected(tmp_path):
    root = Path(__file__).resolve().parents[1] / "protocol"
    path = tmp_path / "study-v1.toml"
    path.write_bytes((root / "study-v1.toml").read_bytes() + b"\n")
    path.with_suffix(".sha256").write_bytes((root / "study-v1.sha256").read_bytes())
    with pytest.raises(ValueError, match="hash mismatch"):
        load_protocol(path)
