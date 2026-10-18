"""Additional ARCH-library GARCH orders for the separately authorized extension."""

import warnings
from dataclasses import dataclass

import numpy as np
from arch import arch_model

from volatility_research.models import Array, mean_square

ORDERS = {"garch_21": (2, 1), "garch_12": (1, 2)}


@dataclass(frozen=True)
class OrderForecast:
    variance: float | None
    failure: str | None
    warning_count: int = 0
    parameters: tuple[float, ...] = ()


def forecast_order(returns: Array, p: int, q: int) -> OrderForecast:
    if (p, q) not in ORDERS.values():
        raise ValueError("Only the two frozen higher-order candidates are supported")
    mean_square(returns)
    caught: list[warnings.WarningMessage] = []
    try:
        model = arch_model(
            returns * 100.0,
            mean="Zero",
            vol="GARCH",
            p=p,
            o=0,
            q=q,
            dist="normal",
            rescale=False,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fitted = model.fit(
                disp="off",
                show_warning=False,
                update_freq=0,
                tol=1e-8,
                options={"maxiter": 1000},
            )
        count = len(caught)
        if fitted.convergence_flag != 0:
            return OrderForecast(None, "nonconvergence", count)
        params = np.asarray(fitted.params, dtype=float)
        if not np.isfinite(params).all():
            return OrderForecast(None, "nonfinite_parameters", count)
        omega = float(fitted.params["omega"])
        alphas = tuple(float(fitted.params[f"alpha[{i}]"]) for i in range(1, p + 1))
        betas = tuple(float(fitted.params[f"beta[{i}]"]) for i in range(1, q + 1))
        coefficients = (*alphas, *betas)
        ordered = (omega, *coefficients)
        if omega <= 0 or any(c < 0 for c in coefficients) or sum(coefficients) >= 1:
            return OrderForecast(None, "inadmissible_parameters", count, ordered)
        forecasts = np.asarray(fitted.forecast(horizon=1, reindex=False).variance, dtype=float)
        variance = float(forecasts[-1, 0]) / 10000
        if not np.isfinite(variance) or variance <= 0:
            return OrderForecast(None, "invalid_forecast", count, ordered)
        return OrderForecast(variance, None, count, ordered)
    except (ValueError, RuntimeError, ArithmeticError, np.linalg.LinAlgError):
        return OrderForecast(None, "fit_exception", len(caught))
