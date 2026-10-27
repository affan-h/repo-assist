import pytest

from evals.stats import bootstrap_ci, mcnemar_exact, paired_bootstrap_diff


def test_bootstrap_interval_contains_the_mean_and_narrows_with_n():
    low, high = bootstrap_ci([1, 0] * 10)
    assert low < 0.5 < high
    wide = high - low
    low, high = bootstrap_ci([1, 0] * 200)
    assert (high - low) < wide / 3


def test_paired_difference():
    mean, low, high = paired_bootstrap_diff([1, 1, 1, 1, 0] * 8, [0, 1, 0, 1, 0] * 8)
    assert mean == pytest.approx(0.4) and low > 0


def test_mcnemar_exact_matches_binomial():
    # 10 discordant pairs, 9 favour A: two-sided p = 2 * P(X <= 1 | n=10, p=.5) = 22/1024
    a = [True] * 9 + [False] + [True] * 5
    b = [False] * 9 + [True] + [True] * 5
    result = mcnemar_exact(a, b)
    assert (result["a_only"], result["b_only"]) == (9, 1)
    assert result["p_value"] == pytest.approx(22 / 1024)
    assert mcnemar_exact([True, False], [True, False])["p_value"] == 1.0
