"""
uncertainty.py
Sensitivity and uncertainty analysis for the weighted-overlay susceptibility
score — the model must never be presented as an absolute, error-free truth
(project section 17).

Two complementary approaches:
  1. Weight-perturbation sensitivity: vary each weight by +/-10% and +/-20%
     (renormalized), recompute the score, and measure how much the final
     classified risk category shifts. This isolates sensitivity to the
     (somewhat subjective) AHP weighting choices.
  2. Monte Carlo input perturbation: add noise to the standardized input
     layers themselves (representing input-data uncertainty, e.g. satellite
     rainfall retrieval error), rerun many times, and report the standard
     deviation of the resulting score per pixel.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .utils import get_logger

logger = get_logger(__name__)


def perturb_weights(weights: dict[str, float], pct: float, direction: int,
                     factor_name: str) -> dict[str, float]:
    """Perturb a single factor's weight by +/-pct, renormalizing all weights
    to sum to 1 afterward (a change to one weight necessarily changes the
    relative influence of all others)."""
    if factor_name not in weights:
        raise ValueError(f"Factor '{factor_name}' not found in weights dict.")
    new_weights = dict(weights)
    delta = weights[factor_name] * pct * direction
    new_weights[factor_name] = max(0.0, weights[factor_name] + delta)
    total = sum(new_weights.values())
    return {k: v / total for k, v in new_weights.items()}


def weight_sensitivity_analysis(base_weights: dict[str, float],
                                 standardized_factors: dict[str, np.ndarray],
                                 pct_levels: list[float] = (0.10, 0.20)) -> dict:
    """For each factor and each perturbation level (+/-), recompute the
    weighted-overlay score and record the mean absolute pixel-level change
    relative to the base score. Returns a per-factor sensitivity ranking
    plus the full set of perturbed score arrays for mapping.
    """
    from .model import HeavyRainSusceptibilityModel

    base_model = HeavyRainSusceptibilityModel(weights=base_weights)
    base_score = base_model.compute(standardized_factors)

    sensitivity_report = []
    perturbed_scores = {}

    for factor_name in base_weights:
        for pct in pct_levels:
            for direction, label in [(1, f"+{int(pct*100)}%"), (-1, f"-{int(pct*100)}%")]:
                new_weights = perturb_weights(base_weights, pct, direction, factor_name)
                model = HeavyRainSusceptibilityModel(weights=new_weights)
                score = model.compute(standardized_factors)

                diff = np.abs(score - base_score)
                mean_abs_change = float(np.nanmean(diff))
                max_abs_change = float(np.nanmax(diff))

                key = f"{factor_name}_{label}"
                perturbed_scores[key] = score
                sensitivity_report.append({
                    "factor": factor_name, "perturbation": label,
                    "mean_abs_change": mean_abs_change, "max_abs_change": max_abs_change,
                })

    import pandas as pd
    report_df = pd.DataFrame(sensitivity_report).sort_values("mean_abs_change", ascending=False)
    logger.info("Weight sensitivity analysis complete. Most sensitive factor: %s",
                report_df.iloc[0]["factor"] if len(report_df) else "N/A")

    return {"base_score": base_score, "report": report_df, "perturbed_scores": perturbed_scores}


def monte_carlo_uncertainty(weights: dict[str, float],
                             standardized_factors: dict[str, np.ndarray],
                             n_iterations: int = 200, noise_sigma: float = 0.05,
                             random_seed: int = 42) -> dict:
    """Monte Carlo simulation of input-data uncertainty: add Gaussian noise
    (sigma=noise_sigma, representing e.g. satellite rainfall retrieval
    error) to each standardized factor layer independently on each
    iteration, recompute the score, and summarise the resulting
    pixel-wise distribution.

    Returns the mean score (should closely match the deterministic base
    score), the standard deviation ("uncertainty layer"), and the
    coefficient of variation.
    """
    from .model import HeavyRainSusceptibilityModel

    rng = np.random.default_rng(random_seed)
    model = HeavyRainSusceptibilityModel(weights=weights)

    shape = next(iter(standardized_factors.values())).shape
    accum = np.zeros((n_iterations, *shape))

    logger.info("Running Monte Carlo uncertainty analysis: %d iterations, sigma=%.3f",
                n_iterations, noise_sigma)

    for i in range(n_iterations):
        noisy_factors = {}
        for name, arr in standardized_factors.items():
            noise = rng.normal(0, noise_sigma, size=arr.shape)
            noisy_factors[name] = np.clip(arr + noise, 0, 1)
        accum[i] = model.compute(noisy_factors)

    mean_score = np.nanmean(accum, axis=0)
    std_score = np.nanstd(accum, axis=0)
    cv = np.where(mean_score > 1e-9, std_score / mean_score, np.nan)

    return {"mean_score": mean_score, "std_score": std_score, "coefficient_of_variation": cv,
            "n_iterations": n_iterations, "noise_sigma": noise_sigma}
