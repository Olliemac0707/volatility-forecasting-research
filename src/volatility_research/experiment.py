"""Sequential paired forecasts, score ledger and no-clobber result publication."""

import csv
import hashlib
import math
from collections import Counter
from collections.abc import Callable
from datetime import UTC, date, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from volatility_research import __version__
from volatility_research.data import sha256, write_json
from volatility_research.models import Array, GarchForecast, ewma_path, garch_forecast, mean_square
from volatility_research.protocol import Protocol
from volatility_research.statistics import hac_interval, qlike

MODELS = ("historical_63", "ewma_094", "garch_11")
Estimator = Callable[[Array], GarchForecast]


def forecast_rows(
    dates: list[date],
    prices: Array,
    protocol: Protocol,
    *,
    estimator: Estimator = garch_forecast,
) -> list[dict[str, Any]]:
    if len(dates) != len(prices) or len(prices) < 2:
        raise ValueError("Date/price length mismatch or empty prices")
    if not np.isfinite(prices).all() or np.any(prices <= 0):
        raise ValueError("Prices must be positive and finite")
    if any(a >= b for a, b in zip(dates, dates[1:], strict=False)):
        raise ValueError("Dates must be strictly increasing")
    returns = np.diff(np.log(prices))
    ewma = ewma_path(returns, protocol.historical_window, protocol.ewma_decay)
    rows: list[dict[str, Any]] = []
    for target in range(1, len(returns)):
        target_date = dates[target + 1]
        if not protocol.evaluation_start <= target_date <= protocol.evaluation_end:
            continue
        # Returns[target] is the unseen target. The prefix stops BEFORE it.
        if target < protocol.garch_window:
            raise ValueError("Insufficient pre-origin history; evaluation dates cannot be dropped")
        training = returns[target - protocol.garch_window : target].copy()
        estimate = estimator(training)
        predicted = estimate.variance
        if predicted is not None and (not math.isfinite(predicted) or predicted <= 0):
            estimate = GarchForecast(None, "invalid_forecast")
            predicted = None
        fallback = estimate.failure is not None or predicted is None
        garch_value = float(ewma[target - 1])
        if not fallback and predicted is not None:
            garch_value = predicted
        raw = {
            "historical_63": mean_square(returns[target - protocol.historical_window : target]),
            "ewma_094": float(ewma[target - 1]),
            "garch_11": garch_value,
        }
        row: dict[str, Any] = {
            "origin_date": str(dates[target]),
            "target_date": str(target_date),
            "training_start": str(dates[target - protocol.garch_window + 1]),
            "training_end": str(dates[target]),
            "training_count": len(training),
            "realized_return": float(returns[target]),
            "target_squared_return": float(returns[target] ** 2),
            "garch_fallback": fallback,
            "garch_warning_count": estimate.warning_count,
            "garch_failure": estimate.failure or ("missing_forecast" if fallback else ""),
            "training_returns_sha256": hashlib.sha256(training.astype("<f8").tobytes()).hexdigest(),
        }
        for name, value in raw.items():
            row[name] = max(protocol.variance_floor, value)
            row[f"{name}_floored"] = value < protocol.variance_floor
        rows.append(row)
    if len(rows) < protocol.minimum_evaluation_count:
        raise ValueError("Too few evaluation observations for the registered uncertainty estimate")
    return rows


def summarize(rows: list[dict[str, Any]], protocol: Protocol) -> dict[str, Any]:
    target = np.asarray([row["target_squared_return"] for row in rows], dtype=float)
    losses: dict[str, Array] = {}
    scores: dict[str, dict[str, float | int]] = {}
    for name in MODELS:
        forecast = np.asarray([row[name] for row in rows], dtype=float)
        losses[name] = qlike(target, forecast)
        scores[name] = {
            "count": len(rows),
            "mean_qlike": float(np.mean(losses[name])),
            "variance_mse": float(np.mean(np.square(target - forecast))),
            "floor_count": sum(row[f"{name}_floored"] for row in rows),
        }
        for row, loss in zip(rows, losses[name], strict=True):
            row[f"{name}_qlike"] = float(loss)
    failures = sum(row["garch_fallback"] for row in rows)
    fraction = failures / len(rows)
    return {
        "interpretation": "Retrospective forecasts; no allocation backtest or alpha claim.",
        "target_count": len(rows),
        "first_target": rows[0]["target_date"],
        "last_target": rows[-1]["target_date"],
        "scores": scores,
        "primary_contrast": "garch_11 minus ewma_094 QLIKE; negative favours GARCH-with-fallback",
        "primary_hac": hac_interval(losses["garch_11"] - losses["ewma_094"]).to_dict(),
        "garch_fallback_count": failures,
        "garch_fallback_fraction": fraction,
        "garch_failure_reasons": dict(
            Counter(row["garch_failure"] for row in rows if row["garch_fallback"])
        ),
        "model_level_claim_permitted_by_fit_gate": fraction
        <= protocol.maximum_garch_fallback_fraction,
        "fit_gate_note": (
            "Fit-gate eligibility is not evidence of improvement. Scores include every fallback."
        ),
    }


def run_experiment(
    dates: list[date],
    prices: Array,
    protocol: Protocol,
    output: Path,
    *,
    kind: str,
    input_sha256: str,
    estimator: Estimator = garch_forecast,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    started = datetime.now(UTC).isoformat()
    source_root = Path(__file__).parent
    sources = {p.name: sha256(p) for p in sorted(source_root.glob("*.py"))}
    receipt = {
        "status": "started",
        "started_at": started,
        "kind": kind,
        "protocol_sha256": protocol.sha256,
        "input_sha256": input_sha256,
        "package_version": __version__,
        "source_sha256": sources,
        "lock_sha256": sha256(source_root.parents[1] / "uv.lock"),
        "effective_evaluation_start": str(protocol.evaluation_start),
        "effective_evaluation_end": str(protocol.evaluation_end),
        "dependencies": {
            name: version(name)
            for name in ("numpy", "pandas", "scipy", "arch", "exchange-calendars")
        },
    }
    write_json(output / "started.json", receipt)
    try:
        rows = forecast_rows(dates, prices, protocol, estimator=estimator)
        summary = summarize(rows, protocol)
        summary["kind"] = kind
        if kind == "synthetic":
            summary["interpretation"] = "Synthetic development check only; no empirical evidence."
        with (output / "forecast-ledger.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        write_json(output / "summary.json", summary)
        write_json(
            output / "completed.json",
            {
                "status": "completed",
                "completed_at": datetime.now(UTC).isoformat(),
                "protocol_sha256": protocol.sha256,
                "input_sha256": input_sha256,
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
                "recorded_at": datetime.now(UTC).isoformat(),
                "protocol_sha256": protocol.sha256,
            },
        )
        raise
