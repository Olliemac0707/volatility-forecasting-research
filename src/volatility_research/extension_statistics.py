"""Two predefined HAC contrasts with Holm family-wise multiplicity control."""

import math
from typing import Any

from volatility_research.models import Array
from volatility_research.statistics import hac_interval


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    if not p_values or any(not math.isfinite(p) or not 0 <= p <= 1 for p in p_values.values()):
        raise ValueError("Holm adjustment requires finite p-values in [0,1]")
    ordered = sorted(p_values, key=lambda name: (p_values[name], name))
    adjusted: dict[str, float] = {}
    previous = 0.0
    for rank, name in enumerate(ordered):
        previous = max(previous, min(1.0, (len(ordered) - rank) * p_values[name]))
        adjusted[name] = previous
    return adjusted


def contrasts(
    paired_differences: dict[str, Array],
    fallback_fractions: dict[str, float],
    max_fallback: float = 0.05,
) -> dict[str, dict[str, Any]]:
    if set(paired_differences) != {"garch_21", "garch_12"}:
        raise ValueError("Both and only both frozen contrasts are required")
    if len({len(values) for values in paired_differences.values()}) != 1:
        raise ValueError("Both contrasts must use the identical target count")
    results: dict[str, dict[str, Any]] = {}
    raw_p: dict[str, float] = {}
    for name, values in paired_differences.items():
        interval = hac_interval(values)
        se = interval.standard_error
        statistic = interval.mean_difference / se if se > 0 else None
        p_value = math.erfc(abs(statistic) / math.sqrt(2)) if statistic is not None else 1.0
        raw_p[name] = p_value
        results[name] = {
            "contrast": f"{name} minus frozen garch_11",
            "marginal_hac_95": interval.to_dict(),
            "asymptotic_z": statistic,
            "raw_two_sided_p": p_value,
            "candidate_fallback_fraction": fallback_fractions[name],
            "fit_gate_pass": fallback_fractions[name] <= max_fallback,
            "degenerate_hac_se": se == 0,
        }
    for name, adjusted in holm_adjust(raw_p).items():
        result = results[name]
        result["holm_adjusted_p"] = adjusted
        result["exploratory_evidence_of_lower_loss"] = bool(
            adjusted <= 0.05
            and result["marginal_hac_95"]["mean_difference"] < 0
            and result["fit_gate_pass"]
            and not result["degenerate_hac_se"]
        )
    return results
