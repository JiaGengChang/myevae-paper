from collections.abc import Collection, Mapping
from math import ceil

import numpy as np
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.util import Surv

DAYS_PER_MONTH = 365.25 / 12.0
WEEK_DAYS = 7.0
HIGH_RISK_FRACTION = 0.20


def weekly_horizons_days() -> np.ndarray:
    start = 12.0 * DAYS_PER_MONTH
    end = 24.0 * DAYS_PER_MONTH
    n_full_weeks = int((end - start) // WEEK_DAYS)
    horizons = start + np.arange(n_full_weeks + 1, dtype=float) * WEEK_DAYS
    if not np.isclose(horizons[-1], end):
        horizons = np.append(horizons, end)
    return horizons


def top_risk_indices(risk_scores, fraction: float = HIGH_RISK_FRACTION) -> np.ndarray:
    scores = np.asarray(risk_scores, dtype=float)
    if scores.ndim != 1 or len(scores) == 0:
        raise ValueError("Risk scores must be a non-empty one-dimensional array")
    if not np.isfinite(scores).all():
        raise ValueError("Risk scores contain non-finite values")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("High-risk fraction must be in (0, 1]")
    n_high_risk = ceil(len(scores) * fraction)
    return np.argsort(-scores, kind="stable")[:n_high_risk]


def _validated_survival(events, times, label: str) -> tuple[np.ndarray, np.ndarray]:
    event_values = np.asarray(events)
    time_values = np.asarray(times, dtype=float)
    if event_values.ndim != 1 or time_values.ndim != 1:
        raise ValueError(f"{label} events and times must be one-dimensional")
    if len(event_values) == 0 or len(event_values) != len(time_values):
        raise ValueError(f"{label} events and times must have the same non-zero length")
    if event_values.dtype.kind == "O" and any(value is None for value in event_values):
        raise ValueError(f"{label} events contain missing values")
    if not np.isfinite(time_values).all():
        raise ValueError(f"{label} times contain non-finite values")
    return event_values.astype(bool), time_values


def calculate_wiauc(
    train_events,
    train_times_days,
    cohorts: Mapping[str, Mapping[str, object]],
    excluded_cohorts: Collection[str] = (),
) -> dict:
    if not cohorts:
        raise ValueError("At least one external cohort is required")

    excluded_cohorts = set(excluded_cohorts)
    unknown_exclusions = excluded_cohorts.difference(cohorts)
    if unknown_exclusions:
        raise ValueError(
            "Cannot exclude unknown cohort(s): " + ", ".join(sorted(unknown_exclusions))
        )

    train_events, train_times = _validated_survival(
        train_events, train_times_days, "Training"
    )
    validated_cohorts = {}
    all_times = [train_times]
    for name, cohort in cohorts.items():
        events, times = _validated_survival(
            cohort["events"], cohort["times_days"], name
        )
        risk_scores = np.asarray(cohort["risk_scores"], dtype=float)
        if risk_scores.ndim != 1 or len(risk_scores) != len(events):
            raise ValueError(f"{name} risk scores must match its survival observations")
        if not np.isfinite(risk_scores).all():
            raise ValueError(f"{name} risk scores contain non-finite values")
        validated_cohorts[name] = (events, times, risk_scores)
        all_times.append(times)

    time_shift_days = max(0.0, -min(float(np.min(times)) for times in all_times))
    survival_train = Surv.from_arrays(train_events, train_times + time_shift_days)
    horizons_days = weekly_horizons_days()
    shifted_horizons = horizons_days + time_shift_days

    cohort_results = {}
    weighted_auc = 0.0
    total_weight = 0
    for name, (events, times, risk_scores) in validated_cohorts.items():
        survival_test = Surv.from_arrays(events, times + time_shift_days)
        try:
            auc_values, _ = cumulative_dynamic_auc(
                survival_train,
                survival_test,
                risk_scores,
                shifted_horizons,
            )
        except Exception as exc:
            raise ValueError(f"Could not calculate time-dependent AUC for {name}: {exc}") from exc
        auc_values = np.asarray(auc_values, dtype=float)
        if auc_values.shape != horizons_days.shape or not np.isfinite(auc_values).all():
            raise ValueError(f"Time-dependent AUC is undefined for one or more {name} horizons")

        n_high_risk = len(top_risk_indices(risk_scores))
        cohort_weight = n_high_risk**2
        iauc = float(np.mean(auc_values))
        cohort_results[name] = {
            "n_patients": int(len(events)),
            "n_high_risk": int(n_high_risk),
            "cohort_weight": int(cohort_weight),
            "iAUC": iauc,
            "weekly_auc": [float(value) for value in auc_values],
            "included_in_wiauc": name not in excluded_cohorts,
        }
        if name not in excluded_cohorts:
            weighted_auc += iauc * cohort_weight
            total_weight += cohort_weight

    if total_weight == 0:
        raise ValueError("No included cohorts have a non-zero squared high-risk cohort weight")

    return {
        "metric": "time-weighted integrated AUC (wiAUC)",
        "auc_method": "scikit-survival cumulative_dynamic_auc with IPCW censoring weights estimated from the model training split",
        "case_control_definition": "Cumulative/dynamic: events at or before each horizon are cases; patients with observed follow-up beyond the horizon are controls.",
        "time_unit": "days",
        "days_per_month": DAYS_PER_MONTH,
        "weekly_step_days": WEEK_DAYS,
        "horizons_days": [float(value) for value in horizons_days],
        "horizons_months": [float(value / DAYS_PER_MONTH) for value in horizons_days],
        "auc_average": "arithmetic mean across weekly horizons, including both 12- and 24-month endpoints",
        "time_shift_days": float(time_shift_days),
        "risk_score_direction": "higher scores indicate higher risk",
        "high_risk_definition": "top 20% of patients ranked by within-cohort predicted risk; ceil(20% of cohort size), stable input-order tie breaking",
        "cohort_weight_definition": "number of high-risk patients squared",
        "excluded_cohorts_from_wiauc": sorted(excluded_cohorts),
        "cohorts": cohort_results,
        "wiAUC": float(weighted_auc / total_weight),
    }