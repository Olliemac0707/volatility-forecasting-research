"""Explicit free-data acquisition; strict normalization and immutable receipts."""

import csv
import hashlib
import json
from datetime import UTC, date, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np

from volatility_research.models import Array


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def read_prices(path: Path) -> tuple[list[date], Array]:
    dates: list[date] = []
    prices: list[float] = []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["date", "adj_close"]:
            raise ValueError("Expected exactly date,adj_close CSV columns")
        for row in reader:
            dates.append(date.fromisoformat(row["date"]))
            prices.append(float(row["adj_close"]))
    values = np.asarray(prices, dtype=np.float64)
    if len(values) < 2 or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("Prices must contain at least two positive finite observations")
    if any(a >= b for a, b in zip(dates, dates[1:], strict=False)):
        raise ValueError("Dates must be strictly ordered without duplicates")
    return dates, values


def validate_calendar(dates: list[date]) -> None:
    import exchange_calendars as xcals

    calendar = xcals.get_calendar("XNYS", start="2005-01-01", end="2026-01-01")
    expected = [d.date() for d in calendar.sessions_in_range("2005-01-03", "2025-12-31")]
    if dates != expected:
        missing = sorted(set(expected) - set(dates))
        extra = sorted(set(dates) - set(expected))
        raise ValueError(f"Snapshot calendar mismatch: {len(missing)} missing, {len(extra)} extra")


def acquire(output: Path, protocol_sha256: str) -> Path:
    """Preserve provider frame before validating or normalizing it. No fallback source."""
    import yfinance as yf

    output.mkdir(parents=True, exist_ok=False)
    try:
        yf.set_tz_cache_location(str(output / "provider-cache"))
        frame = yf.Ticker("SPY").history(
            start="2005-01-03",
            end="2026-01-01",
            interval="1d",
            auto_adjust=False,
            back_adjust=False,
            repair=False,
            actions=True,
            raise_errors=True,
            timeout=20,
        )
        frame.to_csv(output / "provider-frame.csv")
        if frame.empty or "Adj Close" not in frame:
            raise ValueError("Provider returned no usable adjusted-close data")
        prices = output / "prices.csv"
        with prices.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(["date", "adj_close"])
            for stamp, price in zip(frame.index, frame["Adj Close"], strict=True):
                writer.writerow([stamp.date().isoformat(), repr(float(price))])
        dates, _ = read_prices(prices)
        validate_calendar(dates)
        write_json(
            output / "metadata.json",
            {
                "status": "accepted",
                "kind": "historical",
                "symbol": "SPY",
                "price_field": "Adj Close",
                "provider": "Yahoo Finance via yfinance",
                "provider_version": version("yfinance"),
                "calendar_version": version("exchange-calendars"),
                "acquired_at": datetime.now(UTC).isoformat(),
                "rows": len(dates),
                "start": str(dates[0]),
                "end": str(dates[-1]),
                "protocol_sha256": protocol_sha256,
                "prices_sha256": sha256(prices),
                "provider_frame_sha256": sha256(output / "provider-frame.csv"),
                "limitations": (
                    "Revised adjusted history; not point-in-time, execution data "
                    "or redistribution permission."
                ),
            },
        )
        return prices
    except Exception as error:
        write_json(
            output / "acquisition-failure.json",
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "protocol_sha256": protocol_sha256,
                "recorded_at": datetime.now(UTC).isoformat(),
            },
        )
        raise


def verify_snapshot(prices: Path, protocol_sha256: str) -> tuple[list[date], Array]:
    metadata = json.loads((prices.parent / "metadata.json").read_text())
    if metadata["status"] != "accepted" or metadata["kind"] != "historical":
        raise ValueError("An accepted historical acquisition receipt is required")
    if metadata["protocol_sha256"] != protocol_sha256:
        raise ValueError("Snapshot belongs to a different protocol")
    if metadata["prices_sha256"] != sha256(prices):
        raise ValueError("Normalized price snapshot hash mismatch")
    if metadata["provider_frame_sha256"] != sha256(prices.parent / "provider-frame.csv"):
        raise ValueError("Provider snapshot hash mismatch")
    if metadata["symbol"] != "SPY" or metadata["price_field"] != "Adj Close":
        raise ValueError("Unsupported snapshot series")
    dates, values = read_prices(prices)
    validate_calendar(dates)
    return dates, values
