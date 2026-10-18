"""Explicit synthetic, acquisition and historical commands."""

import argparse
import hashlib
import json
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from volatility_research.data import acquire, sha256, verify_snapshot
from volatility_research.experiment import run_experiment
from volatility_research.protocol import load_protocol


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path("protocol/study-v1.toml"))
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("demo", "acquire", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--output", type=Path, required=True)
        if name == "evaluate":
            command.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    # Check protocol identity before reading or acquiring the study data.
    protocol = load_protocol(args.protocol)
    if args.command == "acquire":
        print(acquire(args.output, protocol.sha256))
        return 0
    if args.command == "demo":
        rng = np.random.Generator(np.random.PCG64(47219))
        n = 1380
        innovations = rng.standard_normal(n)
        variance = 0.0001
        returns = np.zeros(n)
        for i in range(n):
            returns[i] = variance**0.5 * innovations[i]
            variance = 0.000002 + 0.08 * returns[i] ** 2 + 0.90 * variance
        prices = np.concatenate(([100.0], 100.0 * np.exp(np.cumsum(returns))))
        dates = [date(2000, 1, 1) + timedelta(days=i) for i in range(n + 1)]
        synthetic_protocol = replace(
            protocol, evaluation_start=dates[1261], evaluation_end=dates[-1]
        )
        summary = run_experiment(
            dates,
            prices,
            synthetic_protocol,
            args.output,
            kind="synthetic",
            input_sha256=hashlib.sha256(prices.astype("<f8").tobytes()).hexdigest(),
        )
    else:
        if args.output.resolve() != Path("reports/spy-daily-variance-v1").resolve():
            parser.error(
                "Historical evaluation uses the fixed reports/spy-daily-variance-v1 output"
            )
        dates, prices = verify_snapshot(args.data, protocol.sha256)
        summary = run_experiment(
            dates,
            prices,
            protocol,
            args.output,
            kind="historical",
            input_sha256=sha256(args.data),
        )
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
