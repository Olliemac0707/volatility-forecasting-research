"""Audit the completed v1 run without refitting a model or changing its specification."""

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from volatility_research.data import sha256, verify_snapshot, write_json
from volatility_research.protocol import load_protocol


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    run = root / "reports/spy-daily-variance-v1"
    protocol = load_protocol(root / "protocol/study-v1.toml")
    dates, prices = verify_snapshot(root / "data/spy-v1/prices.csv", protocol.sha256)
    returns = np.diff(np.log(prices))
    started = json.loads((run / "started.json").read_text())
    completed = json.loads((run / "completed.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    assert started["kind"] == summary["kind"] == "historical"
    assert completed["status"] == "completed"
    assert started["protocol_sha256"] == completed["protocol_sha256"] == protocol.sha256
    assert sha256(root / "uv.lock") == started["lock_sha256"]
    assert sha256(root / "data/spy-v1/prices.csv") == started["input_sha256"]
    assert started["input_sha256"] == completed["input_sha256"]
    for name, expected in started["source_sha256"].items():
        assert sha256(root / "src/volatility_research" / name) == expected
    for name, key in [("forecast-ledger.csv", "ledger_sha256"), ("summary.json", "summary_sha256")]:
        assert sha256(run / name) == completed[key]
    with (run / "forecast-ledger.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected_dates = [d for d in dates if protocol.evaluation_start <= d <= protocol.evaluation_end]
    assert [r["target_date"] for r in rows] == [str(d) for d in expected_dates]
    assert len(rows) == summary["target_count"] == 1508
    by_date = {str(d): i for i, d in enumerate(dates)}
    fallback_count = 0
    for row in rows:
        price_index = by_date[row["target_date"]]
        return_index = price_index - 1
        training = returns[return_index - protocol.garch_window : return_index]
        assert row["origin_date"] == row["training_end"] == str(dates[price_index - 1])
        assert row["training_start"] == str(dates[price_index - protocol.garch_window])
        assert int(row["training_count"]) == len(training) == 1260
        assert (
            hashlib.sha256(training.astype("<f8").tobytes()).hexdigest()
            == row["training_returns_sha256"]
        )
        observed = returns[return_index]
        assert float(row["realized_return"]) == observed
        assert float(row["target_squared_return"]) == observed**2
        fallback = row["garch_fallback"] == "True"
        fallback_count += fallback
        if fallback:
            assert float(row["garch_11"]) == float(row["ewma_094"])
        for model in ("historical_63", "ewma_094", "garch_11"):
            variance = float(row[model])
            assert math.isfinite(variance) and variance >= protocol.variance_floor
            loss = math.log(variance) + observed**2 / variance
            assert math.isclose(loss, float(row[f"{model}_qlike"]), abs_tol=1e-12)
    assert fallback_count == summary["garch_fallback_count"]
    for model in ("historical_63", "ewma_094", "garch_11"):
        average = math.fsum(float(row[f"{model}_qlike"]) for row in rows) / len(rows)
        assert math.isclose(average, summary["scores"][model]["mean_qlike"], abs_tol=1e-12)
    differences = np.array([float(r["garch_11_qlike"]) - float(r["ewma_094_qlike"]) for r in rows])
    n = len(differences)
    lag = math.floor(4 * (n / 100) ** (2 / 9))
    centered = differences - differences.mean()
    lrv = math.fsum(float(v * v) for v in centered) / n
    for j in range(1, lag + 1):
        covariance = (
            math.fsum(float(a * b) for a, b in zip(centered[j:], centered[:-j], strict=True)) / n
        )
        lrv += 2 * (1 - j / (lag + 1)) * covariance
    se = math.sqrt(lrv / n)
    assert lag == summary["primary_hac"]["lag"]
    assert math.isclose(se, summary["primary_hac"]["standard_error"], abs_tol=1e-12)
    public = root / "docs/results"
    public.mkdir(exist_ok=True)
    # Public copies contain aggregate outcomes and hashes, never raw vendor data.
    for original, target in [
        ("summary.json", "summary-v1.json"),
        ("completed.json", "completion-v1.json"),
        ("started.json", "execution-v1.json"),
    ]:
        payload = (run / original).read_bytes()
        destination = public / target
        if destination.exists():
            assert destination.read_bytes() == payload
        else:
            with destination.open("xb") as stream:
                stream.write(payload)
    audit = {
        "status": "PASS",
        "study_id": "spy-daily-variance-v1",
        "target_count": n,
        "snapshot_rows": len(dates),
        "protocol_sha256": protocol.sha256,
        "calendar_complete": True,
        "training_windows_match_snapshot": True,
        "source_and_lock_hashes_match_execution": True,
        "ledger_and_summary_hashes_match_receipt": True,
        "losses_and_hac_se_independently_reconciled": True,
        "garch_fallback_count": fallback_count,
        "note": "Audit only: no model refit, alternate specification or new performance test.",
    }
    audit_path = public / "audit-v1.json"
    if audit_path.exists():
        assert json.loads(audit_path.read_text()) == audit
    else:
        write_json(audit_path, audit)
    print(f"PASS: {n} targets, input/code/output identities, training windows, scores and HAC SE.")


if __name__ == "__main__":
    main()
