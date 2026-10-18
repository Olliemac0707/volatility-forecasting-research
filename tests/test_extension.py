"""Extension integrity: paired dates, past-only inputs, failures and multiplicity."""

from copy import deepcopy
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm

from volatility_research.experiment import forecast_rows, summarize
from volatility_research.extension import (
    extension_rows,
    extension_summary,
    load_extension,
    run_extension,
)
from volatility_research.extension_statistics import contrasts, holm_adjust
from volatility_research.higher_order import OrderForecast
from volatility_research.models import GarchForecast, mean_square
from volatility_research.protocol import load_protocol

ROOT = Path(__file__).resolve().parents[1]


def sample(prices=None):
    if prices is None:
        rng = np.random.default_rng(143)
        prices = np.r_[100.0, 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 1300)))]
    dates = [date(2000, 1, 1) + timedelta(days=i) for i in range(len(prices))]
    protocol = replace(
        load_protocol(ROOT / "protocol/study-v1.toml"),
        evaluation_start=dates[1261],
        evaluation_end=dates[-1],
    )
    rows = forecast_rows(
        dates, prices, protocol, estimator=lambda values: GarchForecast(mean_square(values), None)
    )
    summarize(rows, protocol)
    return dates, prices, rows


def fixed(values, p, q):
    return OrderForecast(mean_square(values) * (1 + p / 10 + q / 20), None)


def test_extension_cannot_see_target_or_future():
    dates, prices, baseline = sample()
    original = extension_rows(dates, prices, baseline, estimator=fixed)
    changed = prices.copy()
    changed[1280:] *= 1.5
    new_dates, changed, changed_baseline = sample(changed)
    later = extension_rows(new_dates, changed, changed_baseline, estimator=fixed)
    for a, b in zip(original, later, strict=True):
        if a["target_date"] <= str(dates[1280]):
            assert a["garch_21"] == b["garch_21"]
            assert a["garch_12"] == b["garch_12"]
            assert a["training_returns_sha256"] == b["training_returns_sha256"]
    assert original[19]["target_squared_return"] != later[19]["target_squared_return"]


def test_exact_pairing_and_baseline_is_never_mutated():
    dates, prices, baseline = sample()
    before = deepcopy(baseline)
    seen = []

    def capture(values, p, q):
        seen.append((values.copy(), p, q))
        return fixed(values, p, q)

    rows = extension_rows(dates, prices, baseline, estimator=capture)
    summary = extension_summary(rows)
    assert baseline == before
    assert summary["target_count"] == 40
    assert len(seen) == 80
    np.testing.assert_allclose(seen[0][0], np.diff(np.log(prices))[:1260])
    np.testing.assert_allclose(seen[2][0], np.diff(np.log(prices))[1:1261])
    assert seen[0][1:] == (2, 1) and seen[1][1:] == (1, 2)
    assert all(row["training_count"] == 1260 for row in rows)
    assert [r["target_date"] for r in rows] == [r["target_date"] for r in baseline]


def test_candidate_failure_keeps_dates_and_does_not_remove_other_model():
    dates, prices, baseline = sample()

    def fail_one(values, p, q):
        return OrderForecast(None, "nonconvergence") if p == 2 else fixed(values, p, q)

    rows = extension_rows(dates, prices, baseline, estimator=fail_one)
    assert len(rows) == 40
    assert all(r["garch_21"] == r["frozen_ewma"] for r in rows)
    assert all(not r["garch_12_fallback"] for r in rows)
    report = extension_summary(rows)
    assert report["failures"]["garch_21"]["fallback_fraction"] == 1
    assert not report["contrasts"]["garch_21"]["fit_gate_pass"]
    assert not report["contrasts"]["garch_21"]["exploratory_evidence_of_lower_loss"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 0])
def test_invalid_forecast_uses_fallback(value):
    dates, prices, baseline = sample()
    rows = extension_rows(
        dates, prices, baseline, estimator=lambda *args: OrderForecast(value, None)
    )
    assert all(r["garch_21_failure"] == r["garch_12_failure"] == "invalid_forecast" for r in rows)


def test_training_hash_mismatch_stops_before_model_fit():
    dates, prices, baseline = sample()
    baseline[0]["training_returns_sha256"] = "invalid"

    def must_not_fit(*args):
        pytest.fail("Invalid baseline must be rejected before candidate fitting")

    with pytest.raises(ValueError, match="window mismatch"):
        extension_rows(dates, prices, baseline, estimator=must_not_fit)


def test_holm_adjustment_accounts_for_both_contrasts():
    assert holm_adjust({"a": 0.03, "b": 0.04}) == {"a": 0.06, "b": 0.06}
    assert holm_adjust({"a": 0.04, "b": 0.01}) == {"b": 0.02, "a": 0.04}
    assert holm_adjust({"a": 0.8, "b": 0.9}) == {"a": 1, "b": 1}


def test_normal_tail_matches_scipy_and_direction_is_required():
    wiggle = np.sin(np.arange(100)) * 0.05
    result = contrasts(
        {"garch_21": wiggle - 0.2, "garch_12": wiggle + 0.2}, {"garch_21": 0, "garch_12": 0}
    )
    for value in result.values():
        assert value["raw_two_sided_p"] == pytest.approx(2 * norm.sf(abs(value["asymptotic_z"])))
    assert result["garch_21"]["exploratory_evidence_of_lower_loss"]
    assert not result["garch_12"]["exploratory_evidence_of_lower_loss"]


def test_zero_hac_se_does_not_create_significance():
    result = contrasts(
        {"garch_21": np.zeros(100), "garch_12": np.full(100, -0.25)}, {"garch_21": 0, "garch_12": 0}
    )
    assert all(r["raw_two_sided_p"] == 1 for r in result.values())
    assert all(not r["exploratory_evidence_of_lower_loss"] for r in result.values())


def test_incomplete_comparison_family_is_rejected():
    with pytest.raises(ValueError):
        contrasts({"garch_21": np.zeros(100)}, {"garch_21": 0})


def test_protocol_change_is_rejected(tmp_path):
    original = ROOT / "protocol/higher-order-v1.toml"
    path = tmp_path / original.name
    path.write_bytes(original.read_bytes() + b"\n")
    path.with_suffix(".sha256").write_bytes(original.with_suffix(".sha256").read_bytes())
    with pytest.raises(ValueError, match="hash mismatch"):
        load_extension(path)


def test_extension_outputs_are_exclusive_and_failures_retained(tmp_path):
    dates, prices, baseline = sample()
    config = load_extension(ROOT / "protocol/higher-order-v1.toml")
    output = tmp_path / "run"
    run_extension(dates, prices, baseline, config, output, kind="synthetic", estimator=fixed)
    assert (output / "completed.json").is_file()
    with pytest.raises(FileExistsError):
        run_extension(dates, prices, baseline, config, output, kind="synthetic", estimator=fixed)
    with pytest.raises(ValueError):
        run_extension(
            dates,
            prices,
            baseline[:2],
            config,
            tmp_path / "failure",
            kind="synthetic",
            estimator=fixed,
        )
    assert (tmp_path / "failure/failed.json").exists()
