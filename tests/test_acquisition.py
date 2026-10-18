"""Exercise the real acquisition adapter with a fake provider, never a network call."""

import json
from types import SimpleNamespace

import pandas as pd
import pytest
import yfinance

from volatility_research import data


@pytest.fixture
def provider(monkeypatch):
    observed = {}
    frame = pd.DataFrame(
        {"Close": [100.0, 102.0, 101.0], "Adj Close": [98.0, 100.0, 99.0], "Volume": [10, 20, 30]},
        index=pd.date_range("2020-01-02", periods=3, tz="America/New_York"),
    )

    def history(**kwargs):
        observed.update(kwargs)
        return frame

    monkeypatch.setattr(yfinance, "Ticker", lambda symbol: SimpleNamespace(history=history))
    monkeypatch.setattr(yfinance, "set_tz_cache_location", lambda path: None)
    monkeypatch.setattr(data, "validate_calendar", lambda dates: None)
    return observed


def test_acquisition_preserves_raw_data_and_exact_adjusted_series(tmp_path, provider):
    prices = data.acquire(tmp_path / "snapshot", "protocol-test")
    dates, values = data.verify_snapshot(prices, "protocol-test")
    assert list(values) == [98.0, 100.0, 99.0]
    assert str(dates[0]) == "2020-01-02"
    assert not provider["auto_adjust"] and not provider["back_adjust"] and not provider["repair"]
    receipt = json.loads((prices.parent / "metadata.json").read_text())
    assert receipt["rows"] == 3
    assert receipt["prices_sha256"] == data.sha256(prices)
    assert receipt["provider_frame_sha256"] == data.sha256(prices.parent / "provider-frame.csv")
    with pytest.raises(FileExistsError):
        data.acquire(prices.parent, "protocol-test")


@pytest.mark.parametrize("filename", ["prices.csv", "provider-frame.csv"])
def test_snapshot_changes_are_rejected(tmp_path, provider, filename):
    prices = data.acquire(tmp_path / "snapshot", "protocol-test")
    changed = prices.parent / filename
    changed.write_text(changed.read_text() + "\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        data.verify_snapshot(prices, "protocol-test")


def test_receipt_cannot_be_reused_for_another_protocol(tmp_path, provider):
    prices = data.acquire(tmp_path / "snapshot", "protocol-test")
    with pytest.raises(ValueError, match="different protocol"):
        data.verify_snapshot(prices, "other-protocol")


def test_failed_acquisition_retains_receipt_without_accepted_snapshot(tmp_path, monkeypatch):
    def unavailable(**kwargs):
        raise RuntimeError("Provider unavailable")

    monkeypatch.setattr(yfinance, "Ticker", lambda symbol: SimpleNamespace(history=unavailable))
    monkeypatch.setattr(yfinance, "set_tz_cache_location", lambda path: None)
    out = tmp_path / "snapshot"
    with pytest.raises(RuntimeError):
        data.acquire(out, "protocol-test")
    assert json.loads((out / "acquisition-failure.json").read_text())["status"] == "failed"
    assert not (out / "metadata.json").exists()
