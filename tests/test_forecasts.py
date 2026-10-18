"""Timing, failures, scaling and estimator checks on synthetic observations."""

from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from volatility_research.experiment import forecast_rows, run_experiment, summarize
from volatility_research.models import GarchForecast, ewma_path, garch_forecast, mean_square
from volatility_research.protocol import load_protocol

ROOT = Path(__file__).resolve().parents[1]


def sample():
    rng = np.random.default_rng(123)
    prices = np.r_[100.0, 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 1320)))]
    dates = [date(2000, 1, 1) + timedelta(days=i) for i in range(len(prices))]
    protocol = replace(
        load_protocol(ROOT / "protocol/study-v1.toml"),
        evaluation_start=dates[1261],
        evaluation_end=dates[-1],
    )
    return dates, prices, protocol


def fixed(returns):
    return GarchForecast(mean_square(returns), None)


def test_forecasts_cannot_see_target_or_future():
    dates, prices, protocol = sample()
    original = forecast_rows(dates, prices, protocol, estimator=fixed)
    changed = prices.copy()
    changed[1280:] *= 1.8
    later = forecast_rows(dates, changed, protocol, estimator=fixed)
    for before, after in zip(original, later, strict=True):
        if before["target_date"] <= str(dates[1280]):
            for model in ("historical_63", "ewma_094", "garch_11"):
                assert before[model] == after[model]
            assert before["training_returns_sha256"] == after["training_returns_sha256"]
    assert original[19]["target_squared_return"] != later[19]["target_squared_return"]


def test_training_window_ends_at_origin_and_has_exact_length():
    dates, prices, protocol = sample()
    seen = []

    def recorder(returns):
        seen.append(returns.copy())
        return fixed(returns)

    rows = forecast_rows(dates, prices, protocol, estimator=recorder)
    assert len(rows) == 60
    np.testing.assert_allclose(seen[0], np.diff(np.log(prices))[:1260])
    np.testing.assert_allclose(seen[1], np.diff(np.log(prices))[1:1261])
    assert all(r["origin_date"] == r["training_end"] < r["target_date"] for r in rows)
    assert all(r["training_count"] == 1260 for r in rows)


def test_failures_retain_all_dates_and_fallback_to_same_origin_ewma():
    dates, prices, protocol = sample()
    rows = forecast_rows(
        dates, prices, protocol, estimator=lambda _: GarchForecast(None, "nonconvergence")
    )
    assert len(rows) == 60
    assert all(r["garch_11"] == r["ewma_094"] and r["garch_fallback"] for r in rows)
    summary = summarize(rows, protocol)
    assert summary["garch_fallback_fraction"] == 1
    assert not summary["model_level_claim_permitted_by_fit_gate"]
    assert summary["primary_hac"]["mean_difference"] == 0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0, -1])
def test_invalid_forecasts_fallback(value):
    dates, prices, protocol = sample()
    rows = forecast_rows(dates, prices, protocol, estimator=lambda _: GarchForecast(value, None))
    assert all(r["garch_failure"] == "invalid_forecast" for r in rows)


def test_ewma_seed_and_update_are_zero_mean_squared_returns():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    actual = ewma_path(values, seed_window=2, decay=0.5)
    assert np.isnan(actual[0])
    np.testing.assert_allclose(actual[1:], [2.5, 5.75, 10.875])
    assert mean_square(np.array([2.0, 2.0])) == 4.0


def test_garch_forecast_is_decimal_variance_and_positive():
    returns = np.random.default_rng(99).normal(0, 0.01, 1260)
    result = garch_forecast(returns)
    assert result.failure is None
    assert 1e-6 < result.variance < 0.001


def test_insufficient_warmup_fails_instead_of_dropping_dates():
    dates, prices, protocol = sample()
    protocol = replace(protocol, evaluation_start=dates[1200])
    with pytest.raises(ValueError, match="history"):
        forecast_rows(dates, prices, protocol, estimator=fixed)


def test_floor_is_applied_and_counted():
    dates, prices, protocol = sample()
    prices[:] = 100
    rows = forecast_rows(dates, prices, protocol, estimator=lambda _: GarchForecast(1e-15, None))
    assert all(r["garch_11"] == protocol.variance_floor for r in rows)
    summary = summarize(rows, protocol)
    assert summary["scores"]["garch_11"]["floor_count"] == 60


def test_output_preserves_existing_run_and_has_hash_receipt(tmp_path):
    dates, prices, protocol = sample()
    out = tmp_path / "run"
    result = run_experiment(
        dates,
        prices,
        protocol,
        out,
        kind="synthetic",
        input_sha256="synthetic-test",
        estimator=fixed,
    )
    assert result["target_count"] == 60
    assert (out / "completed.json").is_file()
    assert len((out / "forecast-ledger.csv").read_text().splitlines()) == 61
    with pytest.raises(FileExistsError):
        run_experiment(
            dates,
            prices,
            protocol,
            out,
            kind="synthetic",
            input_sha256="synthetic-test",
            estimator=fixed,
        )


def test_failed_run_retains_receipt(tmp_path):
    dates, prices, protocol = sample()
    with pytest.raises(ValueError):
        run_experiment(
            dates[:10],
            prices[:10],
            protocol,
            tmp_path / "failed",
            kind="synthetic",
            input_sha256="test",
            estimator=fixed,
        )
    assert (tmp_path / "failed" / "started.json").is_file()
    assert (tmp_path / "failed" / "failed.json").is_file()
    assert not (tmp_path / "failed" / "completed.json").exists()
