"""Uncertainty for small evaluation sets: bootstrap intervals and paired significance tests.

With a few dozen questions, a 10-point difference between two systems can be noise.
Every headline number is reported with a 95% bootstrap interval over questions, and
pairs of systems are compared on the *same* questions (paired tests), which is far
more sensitive than comparing two independent averages.
"""

import math

import numpy as np

N_BOOTSTRAP = 5000
SEED = 7


def bootstrap_ci(values: list[float], alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean."""
    data = np.asarray(values, dtype=float)
    if len(data) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(SEED)
    means = rng.choice(data, size=(N_BOOTSTRAP, len(data)), replace=True).mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_bootstrap_diff(a: list[float], b: list[float], alpha: float = 0.05) -> tuple[float, float, float]:
    """Mean of (a - b) over paired questions, with a bootstrap interval."""
    diffs = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    low, high = bootstrap_ci(list(diffs), alpha)
    return float(diffs.mean()), low, high


def mcnemar_exact(a_correct: list[bool], b_correct: list[bool]) -> dict:
    """Exact McNemar test on paired binary outcomes.

    Only discordant questions matter: `a_only` that A got right and B wrong, and
    vice versa. Under the null hypothesis each discordant pair is a fair coin flip.
    """
    a_only = sum(1 for x, y in zip(a_correct, b_correct, strict=True) if x and not y)
    b_only = sum(1 for x, y in zip(a_correct, b_correct, strict=True) if y and not x)
    n = a_only + b_only
    if n == 0:
        return {"a_only": 0, "b_only": 0, "p_value": 1.0}
    tail = sum(math.comb(n, k) for k in range(min(a_only, b_only) + 1)) / 2 ** n
    return {"a_only": a_only, "b_only": b_only, "p_value": min(1.0, 2 * tail)}
