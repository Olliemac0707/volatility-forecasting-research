"""Separate frozen higher-order extension; the completed baseline is never refitted."""

import argparse
import csv
import hashlib
import json
import math
import tomllib
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from volatility_research.data import sha256, verify_snapshot, write_json
from volatility_research.experiment import forecast_rows, summarize
from volatility_research.extension_statistics import contrasts
from volatility_research.higher_order import ORDERS, OrderForecast, forecast_order
from volatility_research.models import Array
from volatility_research.protocol import load_protocol
from volatility_research.statistics import qlike

ROOT = Path(__file__).resolve().parents[2]
Estimator = Callable[[Array, int, int], OrderForecast]


@dataclass(frozen=True)
class ExtensionProtocol:
    digest: str
    baseline_protocol: str
    snapshot: str
    ledger: str
    summary: str
    completion: str
    lock: str


def load_extension(path: Path) -> ExtensionProtocol:
    digest = sha256(path)
    if path.with_suffix(".sha256").read_text().strip() != digest:
        raise ValueError("Extension protocol hash mismatch")
    values = tomllib.loads(path.read_text())
    if values["study_id"] != "higher-order-garch-v1":
        raise ValueError("Unsupported extension identity")
    if values["candidate_models"] != list(ORDERS):
        raise ValueError("Unsupported extension candidates")
    return ExtensionProtocol(
        digest,
        values["baseline_protocol_sha256"],
        values["snapshot_prices_sha256"],
        values["baseline_ledger_sha256"],
        values["baseline_summary_sha256"],
        values["baseline_completion_sha256"],
        values["dependency_lock_sha256"],
    )


def bound_baseline(
    root: Path, config: ExtensionProtocol
) -> tuple[list[date], Array, list[dict[str, str]]]:
    base = root / "reports/spy-daily-variance-v1"
    identities = {
        root / "protocol/study-v1.toml": config.baseline_protocol,
        root / "data/spy-v1/prices.csv": config.snapshot,
        base / "forecast-ledger.csv": config.ledger,
        base / "summary.json": config.summary,
        base / "completed.json": config.completion,
        root / "uv.lock": config.lock,
    }
    if any(sha256(path) != expected for path, expected in identities.items()):
        raise ValueError("Bound baseline artifact changed")
    protocol = load_protocol(root / "protocol/study-v1.toml")
    dates, prices = verify_snapshot(root / "data/spy-v1/prices.csv", protocol.sha256)
    started = json.loads((base / "started.json").read_text())
    completed = json.loads((base / "completed.json").read_text())
    if completed["status"] != "completed":
        raise ValueError("Completed baseline required")
    for name, expected in started["source_sha256"].items():
        if sha256(root / "src/volatility_research" / name) != expected:
            raise ValueError("Original numerical source changed")
    with (base / "forecast-ledger.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected = [str(d) for d in dates if protocol.evaluation_start <= d <= protocol.evaluation_end]
    if [row["target_date"] for row in rows] != expected or len(rows) != 1508:
        raise ValueError("Original target calendar changed")
    return dates, prices, rows


def extension_rows(
    dates: list[date],
    prices: Array,
    baseline: list[dict[str, Any]],
    *,
    estimator: Estimator = forecast_order,
) -> list[dict[str, Any]]:
    if len(dates) != len(prices) or not np.isfinite(prices).all() or np.any(prices <= 0):
        raise ValueError("Invalid prices")
    if any(a >= b for a, b in zip(dates, dates[1:], strict=False)):
        raise ValueError("Dates must be strictly increasing")
    returns = np.diff(np.log(prices))
    by_date = {str(d): i for i, d in enumerate(dates)}
    target_dates = [str(row["target_date"]) for row in baseline]
    if target_dates != sorted(set(target_dates)):
        raise ValueError("Baseline targets must be unique and strictly increasing")
    output: list[dict[str, Any]] = []
    for original in baseline:
        i = by_date[str(original["target_date"])] - 1
        if i < 1260:
            raise ValueError("Insufficient training history")
        training = returns[i - 1260 : i].copy()
        digest = hashlib.sha256(training.astype("<f8").tobytes()).hexdigest()
        if digest != original["training_returns_sha256"]:
            raise ValueError("Baseline training window mismatch")
        if str(dates[i]) != original["origin_date"]:
            raise ValueError("Baseline origin mismatch")
        target = float(returns[i] ** 2)
        if target != float(original["target_squared_return"]):
            raise ValueError("Baseline realized target mismatch")
        row: dict[str, Any] = {
            "origin_date": original["origin_date"],
            "target_date": original["target_date"],
            "training_start": str(dates[i - 1259]),
            "training_end": str(dates[i]),
            "training_count": len(training),
            "training_returns_sha256": digest,
            "target_squared_return": target,
            "garch_11": float(original["garch_11"]),
            "frozen_ewma": float(original["ewma_094"]),
            "garch_11_qlike": float(original["garch_11_qlike"]),
        }
        for name, (p, q) in ORDERS.items():
            fit = estimator(training.copy(), p, q)
            value = fit.variance
            reason = fit.failure
            if value is not None and (not math.isfinite(value) or value <= 0):
                value, reason = None, "invalid_forecast"
            fallback = reason is not None or value is None
            raw = row["frozen_ewma"] if fallback else value
            assert raw is not None
            row[name] = max(1e-12, raw)
            row[f"{name}_floored"] = raw < 1e-12
            row[f"{name}_fallback"] = fallback
            row[f"{name}_failure"] = reason or ("missing_forecast" if fallback else "")
            row[f"{name}_warning_count"] = fit.warning_count
            row[f"{name}_parameters_percent_units"] = json.dumps(fit.parameters, allow_nan=False)
        output.append(row)
    if len(output) < 30:
        raise ValueError("At least 30 paired observations required")
    return output


def extension_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    target = np.array([row["target_squared_return"] for row in rows], dtype=float)
    losses: dict[str, Array] = {}
    scores: dict[str, Any] = {}
    failures: dict[str, Any] = {}
    fractions: dict[str, float] = {}
    for name in ("garch_11", *ORDERS):
        forecasts = np.array([row[name] for row in rows], dtype=float)
        losses[name] = qlike(target, forecasts)
        if name == "garch_11":
            if not np.allclose(
                losses[name], [row["garch_11_qlike"] for row in rows], atol=1e-12, rtol=0
            ):
                raise ValueError("Frozen baseline loss mismatch")
        scores[name] = {
            "count": len(rows),
            "mean_qlike": float(losses[name].mean()),
            "variance_mse": float(np.mean(np.square(target - forecasts))),
        }
        for row, loss in zip(rows, losses[name], strict=True):
            row[f"{name}_qlike"] = float(loss)
        if name in ORDERS:
            count = sum(row[f"{name}_fallback"] for row in rows)
            fractions[name] = count / len(rows)
            failures[name] = {
                "fallback_count": count,
                "fallback_fraction": fractions[name],
                "floor_count": sum(row[f"{name}_floored"] for row in rows),
                "reasons": dict(
                    Counter(row[f"{name}_failure"] for row in rows if row[f"{name}_fallback"])
                ),
            }
    return {
        "classification": "EXPLORATORY_POST_BASELINE_HISTORICAL_EXTENSION",
        "target_count": len(rows),
        "first_target": rows[0]["target_date"],
        "last_target": rows[-1]["target_date"],
        "scores": scores,
        "failures": failures,
        "contrasts": contrasts(
            {name: losses[name] - losses["garch_11"] for name in ORDERS}, fractions
        ),
        "multiplicity": "Holm across both predefined two-sided contrasts; family alpha=0.05.",
        "interval_note": "95% intervals are marginal; family decisions use Holm-adjusted p-values.",
        "limitations": (
            "Known baseline and reused historical sample. No fresh confirmation, "
            "causal mechanism, alpha or allocation claim."
        ),
    }


def run_extension(
    dates: list[date],
    prices: Array,
    baseline: list[dict[str, Any]],
    config: ExtensionProtocol,
    output: Path,
    *,
    kind: str,
    estimator: Estimator = forecast_order,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).parent
    write_json(
        output / "started.json",
        {
            "status": "started",
            "kind": kind,
            "started_at": datetime.now(UTC).isoformat(),
            "extension_protocol_sha256": config.digest,
            "baseline_ledger_sha256": config.ledger if kind == "historical" else None,
            "input_sha256": config.snapshot
            if kind == "historical"
            else hashlib.sha256(prices.astype("<f8").tobytes()).hexdigest(),
            "source_sha256": {path.name: sha256(path) for path in sorted(source.glob("*.py"))},
            "lock_sha256": sha256(ROOT / "uv.lock"),
            "dependencies": {name: version(name) for name in ("arch", "numpy", "scipy", "pandas")},
        },
    )
    try:
        rows = extension_rows(dates, prices, baseline, estimator=estimator)
        summary = extension_summary(rows)
        summary["kind"] = kind
        if kind == "synthetic":
            summary["classification"] = "SYNTHETIC_DEVELOPMENT_ONLY"
        with (output / "forecast-ledger.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        write_json(output / "summary.json", summary)
        write_json(
            output / "completed.json",
            {
                "status": "completed",
                "completed_at": datetime.now(UTC).isoformat(),
                "extension_protocol_sha256": config.digest,
                "ledger_sha256": sha256(output / "forecast-ledger.csv"),
                "summary_sha256": sha256(output / "summary.json"),
            },
        )
        return summary
    except Exception as error:
        write_json(
            output / "failed.json",
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "extension_protocol_sha256": config.digest,
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_extension(ROOT / "protocol/higher-order-v1.toml")
    if args.demo:
        if args.output is None:
            parser.error("Synthetic runs require an explicit fresh --output directory")
        rng = np.random.default_rng(20261018)
        prices = np.r_[100.0, 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 1300)))]
        dates = [date(2000, 1, 1) + timedelta(days=i) for i in range(len(prices))]
        protocol = replace(
            load_protocol(ROOT / "protocol/study-v1.toml"),
            evaluation_start=dates[1261],
            evaluation_end=dates[-1],
        )
        baseline = forecast_rows(dates, prices, protocol)
        summarize(baseline, protocol)
        result = run_extension(dates, prices, baseline, config, args.output, kind="synthetic")
    else:
        output = ROOT / "reports/higher-order-garch-v1"
        if args.output is not None and args.output.resolve() != output.resolve():
            parser.error("Historical extension uses fixed reports/higher-order-garch-v1 output")
        dates, prices, baseline_rows = bound_baseline(ROOT, config)
        result = run_extension(dates, prices, baseline_rows, config, output, kind="historical")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
