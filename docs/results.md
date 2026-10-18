# Daily variance forecasting: GARCH versus simple baselines

Completed retrospective study · 18 October 2026

## Research question

Does a Gaussian, zero-mean GARCH(1,1) forecast produce lower one-session variance-prediction loss than a fixed EWMA baseline on SPY? This experiment separates a statistical prediction question from a future allocation question. It does not calculate portfolio returns or claim a tradable edge.

The specification was frozen before this project's data acquisition and inspection of model losses. The evaluation years are familiar historical periods, so the study is retrospective rather than genuinely prospective evidence. A freeze does not erase prior knowledge of markets or the choice of models.

## Data and timing

The accepted Yahoo Finance snapshot contains 5,283 adjusted closes from 3 January 2005 through 31 December 2025. Acquisition used pinned yfinance 1.5.1 with automatic adjustment, back adjustment and repair disabled; the explicit `Adj Close` series supplied returns. The complete provider frame and normalized series were preserved separately, hashed, and checked against every expected XNYS session. No missing dates were filled or silently dropped.

Let `r[t] = log(P[t] / P[t−1])`. A forecast issued after close t predicts `r[t+1]^2`. Its inputs stop at the origin close. A daily squared return is a noisy proxy for conditional variance under the declared zero-mean assumption, not an observation of true latent variance. Adjusted histories can be revised and are not a point-in-time feed.

The fixed evaluation interval was 2 January 2020 through 31 December 2025, producing 1,508 paired target dates. Earlier observations supplied initialization and estimation history.

## Fixed model comparison

- **Historical variance:** mean squared return over the most recent 63 sessions, without demeaning under the zero-mean assumption.
- **EWMA:** decay 0.94, seeded from the first 63 squared returns, then recursively updated using only observed returns.
- **GARCH(1,1):** Gaussian innovations and zero mean, fitted afresh at every origin to the most recent 1,260 returns. Returns were scaled to percentage units for fitting, with forecast variances converted back to decimal-return units.

The protocol accepted GARCH fits only when convergence, finite forecasts and stationary nonnegative variance parameters passed their checks. Unacceptable fits would retain the target date and use same-origin EWMA, with an explicit failure reason. This makes the operational comparison a GARCH-with-fallback pipeline versus EWMA. No fallback was required in the realized run.

The numerical variance floor was 1e-12. It was never invoked for any model.

## Primary and secondary outcomes

The primary score was `QLIKE = log(v_hat) + r^2 / v_hat`, where `v_hat` is the strictly positive variance forecast. Lower is better. The score can be negative and is not an investment return or a percentage improvement.

The primary contrast was the paired daily GARCH-minus-EWMA loss difference. Uncertainty used a two-sided normal 95% interval with Newey-West Bartlett HAC standard error. The registered lag rule selected seven lags for this sample. All-model scores and variance MSE were secondary descriptive outputs, not alternate ways to declare success.

| Model | Target count | Mean QLIKE | Variance MSE |
|---|---:|---:|---:|
| Historical, 63 sessions | 1,508 | −7.956398 | 4.457867 × 10⁻⁷ |
| EWMA, decay 0.94 | 1,508 | −8.100473 | 3.774028 × 10⁻⁷ |
| GARCH(1,1) | 1,508 | −8.144203 | 3.253077 × 10⁻⁷ |

MSE is measured in squared variance units (the fourth power of decimal-return units). It is a secondary descriptive score; its ordering does not replace the primary comparison.

| Registered primary comparison | Value |
|---|---:|
| Mean GARCH-minus-EWMA QLIKE | −0.0437292009961 |
| Newey-West standard error | 0.0224080416396 |
| HAC lag | 7 |
| 95% lower bound | −0.0876481555738 |
| 95% upper bound | +0.0001897535817 |
| GARCH fallback fits | 0 of 1,508 |

![Model loss comparison and registered primary interval](figures/forecast-comparison.svg)

**Interpretation:** GARCH had lower observed average QLIKE than EWMA, but the registered two-sided 95% interval includes zero. The experiment therefore does not establish lower expected loss under this uncertainty procedure. It also does not establish equivalence or the absence of any forecasting value. Changing to a one-sided test, another uncertainty estimator or a more favourable sample after seeing this interval would change the registered comparison.

## Execution and independent reconciliation

The forecast evaluation executed once. Its completion receipt binds the input snapshot, protocol, forecast ledger and summary. A subsequent audit checked the stored code and dependency-lock hashes against the execution receipt; every training window against its exact input prefix; every target against the complete exchange-calendar sample; each ledger loss against its recorded forecast and realization; and the mean losses and HAC standard error using independent summations. That audit did not refit any model or introduce a new specification.

The earlier offline validation passed 38 tests, Ruff, strict mypy and dependency checks. Tests cover look-ahead contamination, training boundaries, percentage-to-decimal variance scaling, numerical failures, fallback retention, input tampering, protocol changes and no-clobber outputs. A separate synthetic demo exercised 120 real GARCH forecasts. Neither automated-test counts nor synthetic comparative losses constitute market evidence.

The public evidence copies contain aggregate results and provenance receipts. The detailed forecast ledger and vendor price files remain local and excluded from the source package. Public readers can run the synthetic checks; exact historical reconstruction additionally requires the matching accepted input snapshot. A fresh Yahoo download can have different adjusted values.

## Limitations and stopping point

The result concerns one instrument, one chosen historical interval and three fixed estimators. The HAC interval is asymptotic and conditional on these design choices; nonstationarity and wider research-selection effects remain. No claim of global statistical significance, persistent alpha, executable returns or economic utility follows from these scores.

Better variance predictions do not automatically make a better allocation strategy. Close-based forecasts cannot retroactively determine an overnight holding. An allocation experiment would need its own fixed execution, cost, interest, sizing and benchmark assumptions before examining portfolio outcomes.

This forecast comparison is complete. Its model set, decay, estimation windows, evaluation dates, loss and uncertainty method remain unchanged after the outcome. Further research requires a new question rather than a search for a more favourable version of this result.

## Evidence files

- [Protocol](../protocol/study-v1.toml) and [protocol hash](../protocol/study-v1.sha256).
- [Exact aggregate summary](results/summary-v1.json).
- [Execution receipt](results/execution-v1.json) and [completion receipt](results/completion-v1.json).
- [Independent reconciliation record](results/audit-v1.json).
- [Methodological assumptions and references](research-design.md).
- [Downloadable figure PDF](figures/forecast-comparison.pdf).
