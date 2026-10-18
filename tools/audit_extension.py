"""Reconcile the frozen extension without refitting any historical model."""

import csv
import hashlib
import json
import math

import numpy as np
from scipy.stats import norm

from volatility_research.data import sha256, write_json
from volatility_research.extension import ROOT, bound_baseline, load_extension


def main() -> None:
    config = load_extension(ROOT / "protocol/higher-order-v1.toml")
    dates, prices, baseline = bound_baseline(ROOT, config)
    run = ROOT / "reports/higher-order-garch-v1"
    started = json.loads((run / "started.json").read_text())
    completed = json.loads((run / "completed.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    assert completed["status"] == "completed" and started["kind"] == "historical"
    assert summary["classification"] == "EXPLORATORY_POST_BASELINE_HISTORICAL_EXTENSION"
    assert (
        started["extension_protocol_sha256"]
        == completed["extension_protocol_sha256"]
        == config.digest
    )
    assert started["input_sha256"] == config.snapshot
    assert started["baseline_ledger_sha256"] == config.ledger
    assert started["lock_sha256"] == config.lock == sha256(ROOT / "uv.lock")
    for name, digest in started["source_sha256"].items():
        assert sha256(ROOT / "src/volatility_research" / name) == digest
    for name, key in [("forecast-ledger.csv", "ledger_sha256"), ("summary.json", "summary_sha256")]:
        assert sha256(run / name) == completed[key]
    with (run / "forecast-ledger.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(baseline) == summary["target_count"] == 1508
    returns = np.diff(np.log(prices))
    indices = {str(day): index for index, day in enumerate(dates)}
    fallback_counts = {name: 0 for name in ("garch_21", "garch_12")}
    for row, original in zip(rows, baseline, strict=True):
        assert row["target_date"] == original["target_date"]
        assert row["origin_date"] == row["training_end"] == original["origin_date"]
        i = indices[row["target_date"]] - 1
        training = returns[i - 1260 : i]
        assert row["training_start"] == str(dates[i - 1259])
        assert row["training_count"] == "1260"
        assert row["training_returns_sha256"] == original["training_returns_sha256"]
        assert (
            hashlib.sha256(training.astype("<f8").tobytes()).hexdigest()
            == row["training_returns_sha256"]
        )
        target = float(returns[i] ** 2)
        assert float(row["target_squared_return"]) == target
        assert float(row["garch_11"]) == float(original["garch_11"])
        assert float(row["frozen_ewma"]) == float(original["ewma_094"])
        for name in ("garch_11", "garch_21", "garch_12"):
            forecast = float(row[name])
            assert forecast >= 1e-12 and math.isfinite(forecast)
            loss = math.log(forecast) + target / forecast
            assert math.isclose(loss, float(row[f"{name}_qlike"]), abs_tol=1e-12)
            if name != "garch_11":
                fallback = row[f"{name}_fallback"] == "True"
                fallback_counts[name] += fallback
                if fallback:
                    assert forecast == float(row["frozen_ewma"])
                else:
                    parameters = json.loads(row[f"{name}_parameters_percent_units"])
                    assert len(parameters) == 4 and all(math.isfinite(p) for p in parameters)
                    assert parameters[0] > 0 and min(parameters[1:]) >= 0
                    assert sum(parameters[1:]) < 1
    raw_p = {}
    for name in ("garch_11", "garch_21", "garch_12"):
        average = math.fsum(float(row[f"{name}_qlike"]) for row in rows) / len(rows)
        mse = math.fsum(
            (float(row["target_squared_return"]) - float(row[name])) ** 2 for row in rows
        ) / len(rows)
        assert math.isclose(average, summary["scores"][name]["mean_qlike"], abs_tol=1e-12)
        assert math.isclose(mse, summary["scores"][name]["variance_mse"], rel_tol=1e-12)
        if name == "garch_11":
            continue
        assert fallback_counts[name] == summary["failures"][name]["fallback_count"]
        values = np.array([float(r[f"{name}_qlike"]) - float(r["garch_11_qlike"]) for r in rows])
        n = len(values)
        average = math.fsum(values) / n
        centered = values - average
        lag = math.floor(4 * (n / 100) ** (2 / 9))
        lrv = math.fsum(float(v * v) for v in centered) / n
        for j in range(1, lag + 1):
            cov = (
                math.fsum(float(a * b) for a, b in zip(centered[j:], centered[:-j], strict=True))
                / n
            )
            lrv += 2 * (1 - j / (lag + 1)) * cov
        se = math.sqrt(max(lrv, 0) / n)
        interval = summary["contrasts"][name]["marginal_hac_95"]
        assert math.isclose(average, interval["mean_difference"], abs_tol=1e-12)
        assert math.isclose(se, interval["standard_error"], abs_tol=1e-12)
        assert interval["lag"] == lag
        z95 = float(norm.ppf(0.975))
        assert math.isclose(average - z95 * se, interval["lower_95"], abs_tol=1e-12)
        assert math.isclose(average + z95 * se, interval["upper_95"], abs_tol=1e-12)
        raw_p[name] = float(2 * norm.sf(abs(average / se))) if se > 0 else 1.0
        assert math.isclose(
            raw_p[name], summary["contrasts"][name]["raw_two_sided_p"], abs_tol=1e-12
        )
    ordered = sorted(raw_p, key=lambda name: (raw_p[name], name))
    first, second = ordered
    adjusted = {
        first: min(1, 2 * raw_p[first]),
        second: min(1, max(2 * raw_p[first], raw_p[second])),
    }
    for name, p_value in adjusted.items():
        contrast = summary["contrasts"][name]
        assert math.isclose(p_value, contrast["holm_adjusted_p"], abs_tol=1e-12)
        expected = bool(
            p_value <= 0.05
            and contrast["marginal_hac_95"]["mean_difference"] < 0
            and fallback_counts[name] / len(rows) <= 0.05
            and contrast["marginal_hac_95"]["standard_error"] > 0
        )
        assert contrast["exploratory_evidence_of_lower_loss"] == expected
    public = ROOT / "docs/results/higher-order-v1"
    public.mkdir(parents=True, exist_ok=True)
    for name in ("started.json", "completed.json", "summary.json"):
        payload = (run / name).read_bytes()
        path = public / name
        if path.exists():
            assert path.read_bytes() == payload
        else:
            with path.open("xb") as stream:
                stream.write(payload)
    audit = {
        "status": "PASS",
        "study_id": "higher-order-garch-v1",
        "target_count": len(rows),
        "extension_protocol_sha256": config.digest,
        "original_study_unchanged": True,
        "exact_frozen_baseline_reused": True,
        "windows_targets_and_parameters_validated": True,
        "losses_mse_hac_intervals_and_holm_independently_reconciled": True,
        "fallback_counts": fallback_counts,
        "historical_model_refits_during_audit": 0,
    }
    path = public / "audit.json"
    if path.exists():
        assert json.loads(path.read_text()) == audit
    else:
        write_json(path, audit)
    print(
        "PASS: frozen baseline, 1,508 paired targets, parameters, losses, HAC and Holm decisions."
    )


if __name__ == "__main__":
    main()
