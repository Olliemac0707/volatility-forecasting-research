"""Boundary tests for library units and failed fits, independent of market outcomes."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from volatility_research import models


def install_fit(monkeypatch, *, flag=0, omega=0.02, alpha=0.1, beta=0.85, variance=2.5):
    observed = {}
    fitted = SimpleNamespace(
        convergence_flag=flag,
        params=pd.Series({"omega": omega, "alpha[1]": alpha, "beta[1]": beta}),
        forecast=lambda **kwargs: SimpleNamespace(variance=pd.DataFrame([[variance]])),
    )

    def build(returns, **kwargs):
        observed["returns"] = returns.copy()
        observed["specification"] = kwargs

        def fit(**options):
            observed["fit_options"] = options
            return fitted

        return SimpleNamespace(fit=fit)

    monkeypatch.setattr(models, "arch_model", build)
    return observed


def test_percent_fit_is_converted_back_to_decimal_variance(monkeypatch):
    observed = install_fit(monkeypatch)
    returns = np.array([0.01, -0.02, 0.005])
    result = models.garch_forecast(returns)
    np.testing.assert_allclose(observed["returns"], [1.0, -2.0, 0.5])
    assert result.variance == pytest.approx(0.00025)
    assert result.failure is None
    assert observed["specification"]["mean"] == "Zero"
    assert observed["specification"]["rescale"] is False


@pytest.mark.parametrize(
    "options,reason",
    [
        ({"flag": 1}, "nonconvergence"),
        ({"omega": float("nan")}, "nonfinite_parameters"),
        ({"alpha": 0.2, "beta": 0.8}, "inadmissible_parameters"),
        ({"omega": -0.02}, "inadmissible_parameters"),
        ({"variance": float("nan")}, "invalid_forecast"),
    ],
)
def test_unacceptable_fit_is_not_scored_as_success(monkeypatch, options, reason):
    install_fit(monkeypatch, **options)
    result = models.garch_forecast(np.array([0.01, -0.01]))
    assert result.variance is None and result.failure == reason


def test_numerical_failure_has_explicit_status(monkeypatch):
    def broken(*args, **kwargs):
        raise np.linalg.LinAlgError("Singular test fit")

    monkeypatch.setattr(models, "arch_model", broken)
    result = models.garch_forecast(np.array([0.01, -0.01]))
    assert result.variance is None and result.failure == "fit_exception"
