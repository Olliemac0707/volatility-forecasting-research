"""Check lag notation, all coefficients and variance scaling at the library boundary."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from volatility_research import higher_order


def install(monkeypatch, parameters, *, flag=0):
    seen = {}

    def build(values, **kwargs):
        seen.update(kwargs)
        seen["returns"] = values
        fitted = SimpleNamespace(
            convergence_flag=flag,
            params=pd.Series(parameters),
            forecast=lambda **kw: SimpleNamespace(variance=pd.DataFrame([[2.5]])),
        )
        return SimpleNamespace(fit=lambda **kw: fitted)

    monkeypatch.setattr(higher_order, "arch_model", build)
    return seen


@pytest.mark.parametrize(
    "p,q,parameters",
    [
        (2, 1, {"omega": 0.02, "alpha[1]": 0.05, "alpha[2]": 0.05, "beta[1]": 0.8}),
        (1, 2, {"omega": 0.02, "alpha[1]": 0.1, "beta[1]": 0.7, "beta[2]": 0.1}),
    ],
)
def test_lag_convention_and_percent_variance_conversion(monkeypatch, p, q, parameters):
    seen = install(monkeypatch, parameters)
    result = higher_order.forecast_order(np.array([0.01, -0.02]), p, q)
    assert seen["p"] == p and seen["q"] == q and seen["o"] == 0
    np.testing.assert_allclose(seen["returns"], [1, -2])
    assert result.variance == pytest.approx(0.00025)
    assert result.failure is None and len(result.parameters) == 4


@pytest.mark.parametrize(
    "p,q,parameters",
    [
        (2, 1, {"omega": 0.02, "alpha[1]": 0.1, "alpha[2]": -0.1, "beta[1]": 0.7}),
        (1, 2, {"omega": 0.02, "alpha[1]": 0.1, "beta[1]": 0.7, "beta[2]": -0.1}),
        (2, 1, {"omega": 0.02, "alpha[1]": 0.1, "alpha[2]": 0.3, "beta[1]": 0.7}),
        (1, 2, {"omega": 0.02, "alpha[1]": 0.1, "beta[1]": 0.7, "beta[2]": 0.3}),
    ],
)
def test_second_lag_is_included_in_admissibility_checks(monkeypatch, p, q, parameters):
    install(monkeypatch, parameters)
    result = higher_order.forecast_order(np.array([0.01, -0.02]), p, q)
    assert result.failure == "inadmissible_parameters" and result.variance is None


def test_unsupported_order_cannot_silently_expand_research():
    with pytest.raises(ValueError, match="frozen"):
        higher_order.forecast_order(np.array([0.01, -0.02]), 2, 2)
