# Research design and interpretation

## What is being predicted?

The adjusted close-to-close log return is `r[t] = log(P[t] / P[t-1])`. Forecasts estimate its conditional second moment under the declared zero-mean assumption. The target is `r[t+1]^2`. A nonzero conditional mean would make that second moment differ from conditional variance. Adjusted closes approximate distribution-adjusted histories and can be revised retrospectively.

The forecast is issued after close t. The historical baseline is the mean of the last 63 squared returns, not the demeaned sample variance. EWMA starts with the first 63-return second moment and updates through the issuance return using lambda 0.94. GARCH is estimated afresh on the last 1,260 observations; it sees no target return. All models score the same target dates.

## Why QLIKE?

For positive forecast v, the score `log(v) + r^2/v` penalizes overly small forecasts sharply on high-return-magnitude dates. Lower is better; the score can be negative in decimal-return units. It differs by a target-only term from another common QLIKE convention, so model loss differences are the relevant comparison. MSE is secondary; a more attractive MSE cannot replace the primary QLIKE outcome.

The robustness arguments for variance-proxy comparisons depend on conditions such as an appropriate conditionally unbiased proxy and integrability. A daily squared return is noisy. Neither QLIKE nor the label “realized” turns a daily squared observation into true latent variance.

## What is the primary estimand?

The time average of paired daily QLIKE loss differences for the fixed GARCH-with-EWMA-fallback pipeline minus EWMA. Negative favours the candidate pipeline. The Newey-West interval accounts for short-run serial dependence using the registered Bartlett weights and automatic lag rule. Its coverage is asymptotic and conditional on the specification; it does not correct the entire history of model choice, known market periods or nonstationarity.

An interval including zero is inconclusive for an average-loss difference under this procedure. An interval wholly below zero would support lower mean loss in this historical experiment, not a claim of tradable alpha or future dominance. A fit-failure fraction above 5% disallows a model-level GARCH claim even though every fallback-pipeline score remains reported. Below that threshold, primary results still concern the pipeline, with fit eligibility explicitly distinguished from evidence of improvement.

## Historical knowledge and development

The 2020–2025 period is a retrospective evaluation interval selected before this project's data acquisition and loss inspection. It is familiar history, not a genuinely unseen or prospective test. No model parameters, candidates or primary contrasts will be selected from those losses. The earlier data supplies initialization and rolling estimation; it is not a license to tune a competing specification.

Synthetic development uses a fixed random seed and a clustered-volatility generator. That generator is favourable to the general GARCH model class by construction. It is suitable for engineering checks, not comparative model evidence.

## What this experiment does not establish

No equity curve, investor utility or allocation performance is calculated in phase one. A forecast issued after a close could inform a later order, but cannot retroactively allocate the overnight interval. The future allocation experiment needs a separate timing and cost design; dividing a volatility target by a forecast is not yet an executable historical strategy.

## Method references to consult

- Engle (1982), “Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation,” Econometrica.
- Bollerslev (1986), “Generalized Autoregressive Conditional Heteroskedasticity,” Journal of Econometrics.
- Newey and West (1987), “A Simple, Positive Semi-definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix,” Econometrica.
- Patton (2011), “Volatility forecast comparison using imperfect volatility proxies,” Journal of Econometrics.

These are methodological references, not a claim that every assumption in those papers holds for this data. Read the relevant assumptions before making an empirical inference claim.
