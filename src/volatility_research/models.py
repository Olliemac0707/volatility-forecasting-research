"""Past-only variance estimators with explicit scaling and GARCH failure handling."""

import warnings
from dataclasses import dataclass

import numpy as np
from arch import arch_model
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True)
class GarchForecast:
    variance: float | None
    failure: str | None
    warning_count: int = 0


def mean_square(returns: Array) -> float:
    if returns.ndim != 1 or not len(returns) or not np.isfinite(returns).all():
        raise ValueError("Returns must be a nonempty finite vector")
    return float(np.mean(np.square(returns)))


def ewma_path(returns: Array, seed_window: int = 63, decay: float = 0.94) -> Array:
    """Index i is the forecast issued after observing return i, for return i+1."""
    if seed_window < 1 or not 0 < decay < 1 or len(returns) < seed_window:
        raise ValueError("Invalid EWMA initialization")
    mean_square(returns)
    forecasts = np.full(len(returns), np.nan, dtype=np.float64)
    forecasts[seed_window - 1] = mean_square(returns[:seed_window])
    for i in range(seed_window, len(returns)):
        forecasts[i] = decay * forecasts[i - 1] + (1 - decay) * returns[i] ** 2
    return forecasts


def garch_forecast(returns: Array) -> GarchForecast:
    """Fit percent returns for numerical conditioning; return decimal variance."""
    mean_square(returns)
    try:
        model = arch_model(
            returns * 100.0,
            mean="Zero",
            vol="GARCH",
            p=1,
            o=0,
            q=1,
            dist="normal",
            rescale=False,
        )
        # Capture warnings and record their count without changing the registered
        # convergence/parameter acceptance criteria.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fitted = model.fit(
                disp="off",
                show_warning=False,
                update_freq=0,
                tol=1e-8,
                options={"maxiter": 1000},
            )
        if fitted.convergence_flag != 0:
            return GarchForecast(None, "nonconvergence")
        params = np.asarray(fitted.params, dtype=float)
        if not np.isfinite(params).all():
            return GarchForecast(None, "nonfinite_parameters")
        omega = float(fitted.params["omega"])
        alpha = float(fitted.params["alpha[1]"])
        beta = float(fitted.params["beta[1]"])
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
            return GarchForecast(None, "inadmissible_parameters")
        forecasts = np.asarray(fitted.forecast(horizon=1, reindex=False).variance, dtype=float)
        value = float(forecasts[-1, 0]) / 10000
        if not np.isfinite(value) or value <= 0:
            return GarchForecast(None, "invalid_forecast")
        return GarchForecast(value, None, len(caught))
    except (ValueError, RuntimeError, ArithmeticError, np.linalg.LinAlgError):
        return GarchForecast(None, "fit_exception")
