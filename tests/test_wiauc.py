import numpy as np
import pytest

import utils.wiauc as wiauc


def test_weekly_horizons_cover_12_to_24_months():
    horizons = wiauc.weekly_horizons_days()

    assert horizons[0] == pytest.approx(12 * wiauc.DAYS_PER_MONTH)
    assert horizons[-1] == pytest.approx(24 * wiauc.DAYS_PER_MONTH)
    assert np.all(np.diff(horizons) > 0)
    assert np.allclose(np.diff(horizons)[:-1], wiauc.WEEK_DAYS)


def test_top_risk_indices_uses_stable_order_for_ties():
    scores = np.array([0.9, 0.9, 0.2, 0.5, 0.1, 0.3, 0.4, 0.6, 0.7, 0.8])

    assert wiauc.top_risk_indices(scores).tolist() == [0, 1]


def test_calculate_wiauc_uses_squared_high_risk_counts(monkeypatch):
    def fake_auc(survival_train, survival_test, estimate, times):
        value = float(estimate[0])
        return np.full(len(times), value), value

    monkeypatch.setattr(wiauc, "cumulative_dynamic_auc", fake_auc)
    train_events = np.array([True, False, True, False])
    train_times = np.array([-2.0, 20.0, 80.0, 120.0])
    cohorts = {
        "small": {
            "events": np.array([True] * 5),
            "times_days": np.array([800.0] * 5),
            "risk_scores": np.array([0.6] * 5),
        },
        "large": {
            "events": np.array([True] * 10),
            "times_days": np.array([800.0] * 10),
            "risk_scores": np.array([0.8] * 10),
        },
    }

    result = wiauc.calculate_wiauc(train_events, train_times, cohorts)

    assert result["time_shift_days"] == 2.0
    assert result["cohorts"]["small"]["n_high_risk"] == 1
    assert result["cohorts"]["large"]["n_high_risk"] == 2
    assert result["cohorts"]["small"]["cohort_weight"] == 1
    assert result["cohorts"]["large"]["cohort_weight"] == 4
    assert result["wiAUC"] == pytest.approx((0.6 + 0.8 * 4) / 5)


def test_calculate_wiauc_fails_on_undefined_horizon_auc(monkeypatch):
    def undefined_auc(survival_train, survival_test, estimate, times):
        values = np.ones(len(times))
        values[3] = np.nan
        return values, np.nan

    monkeypatch.setattr(wiauc, "cumulative_dynamic_auc", undefined_auc)
    cohorts = {
        "cohort": {
            "events": np.array([True, False]),
            "times_days": np.array([800.0, 900.0]),
            "risk_scores": np.array([0.9, 0.1]),
        }
    }

    with pytest.raises(ValueError, match="AUC is undefined"):
        wiauc.calculate_wiauc(
            np.array([True, False]), np.array([20.0, 80.0]), cohorts
        )


def test_calculate_wiauc_requires_cohorts():
    with pytest.raises(ValueError, match="At least one external cohort"):
        wiauc.calculate_wiauc(np.array([True]), np.array([10.0]), {})