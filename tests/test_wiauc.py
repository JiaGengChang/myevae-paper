import json
from importlib import import_module
import sys
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

import utils.wiauc as wiauc

score_wiauc = import_module("pipeline.4_score_wiauc")


@pytest.mark.parametrize(
    ("architecture", "feature_name"),
    [
        ("VAE", "full_features_pfs_processed_mut_nan.parquet"),
        ("CoxPH", "full_features_pfs_processed.parquet"),
    ],
)
def test_load_training_data_uses_full_preprocessing_artifacts(
    tmp_path, monkeypatch, architecture, feature_name
):
    features_path = tmp_path / feature_name
    labels_path = tmp_path / "full_labels.parquet"
    features_path.touch()
    labels_path.touch()
    read_paths = []

    def fake_read_parquet(path):
        read_paths.append(path)
        return pd.DataFrame()

    monkeypatch.setenv("SPLITDATADIR", str(tmp_path))
    monkeypatch.setattr(score_wiauc.pd, "read_parquet", fake_read_parquet)

    score_wiauc._load_training_data(
        {
            "endpoint": "pfs",
            "fulldata": True,
            "shuffle": 9,
            "fold": 4,
            "architecture": architecture,
        }
    )

    assert read_paths == [features_path, labels_path]


def test_model_params_fall_back_to_params_fixed_for_full_model():
    params_fixed = {"z_dim": 128, "input_types": ["exp"]}

    assert score_wiauc._model_params({"params_fixed": params_fixed}) is params_fixed


def test_model_architecture_detects_coxph_baseline_metadata():
    assert score_wiauc._model_architecture({"endpoint": "pfs", "use_clin": True}) == "coxph"


def test_model_input_types_reject_external_modalities_not_available():
    with pytest.raises(ValueError, match="support only expression and clinical"):
        score_wiauc._model_input_types(
            "deepsurv",
            {"input_types_all": "['mut', 'clin']"},
            {},
        )


@pytest.mark.parametrize(
    ("architecture", "module_name", "class_name"),
    [
        ("coxnet", "modules_coxnet.estimator", "Coxnet"),
        ("rsf", "modules_rsf.estimator", "RSF"),
    ],
)
def test_build_model_refits_sksurv_estimators_from_saved_params(
    architecture, module_name, class_name, tmp_path, monkeypatch
):
    class FakeEstimator:
        def __init__(self, eventcol, durationcol, input_types_all, subset_microarray=False, **kwargs):
            self.model = object()
            self.input_types_all = input_types_all

        def fit(self, training_data):
            self.training_data = training_data
            return self

    fake_module = ModuleType(module_name)
    setattr(fake_module, class_name, FakeEstimator)
    monkeypatch.setitem(sys.modules, module_name, fake_module)
    weights_file = tmp_path / "model.pth"
    weights_file.write_text(json.dumps({"n_estimators": 17, "l1_ratio": 0.3}))
    results = {
        "params_fixed": {
            "architecture": architecture,
            "endpoint": "pfs",
            "eventcol": "censpfs",
            "durationcol": "pfscdy",
            "subset": False,
        },
        "best_epoch": {
            "params": {
                "input_types_all": "['exp', 'clin']",
                "scale_method": "std",
            }
        },
    }
    train_features = pd.DataFrame({"Feature_exp_G1": [0.1, 0.2]})
    train_labels = pd.DataFrame({"censpfs": [True, False], "pfscdy": [10, 20]})

    model, input_types, subtask_inputs, prediction_type = score_wiauc._build_model(
        results, train_features, train_labels, architecture, weights_file
    )

    assert model is not None
    assert input_types == ["exp", "clin"]
    assert subtask_inputs == []
    assert prediction_type == "sklearn"


def test_build_coxph_model_refits_selected_baseline(tmp_path, monkeypatch):
    class FakeBaseline:
        def fit(self, training_data, survival):
            self.training_data = training_data
            self.survival = survival
            return self

    import utils.coxph as coxph

    captured = {}

    def fake_create_baseline_model(feature_pattern, use_clin):
        captured["feature_pattern"] = feature_pattern
        captured["use_clin"] = use_clin
        return FakeBaseline()

    monkeypatch.setattr(coxph, "create_baseline_model", fake_create_baseline_model)
    monkeypatch.setattr(
        score_wiauc,
        "_external_coxph_inputs",
        lambda endpoint: {
            "UAMS": {
                "features": pd.DataFrame(
                    columns=["Feature_clin_D_PT_age", "Feature_GEP_UAMS70"]
                )
            }
        },
    )
    score_path = tmp_path / "commpass_scores.csv"
    pd.DataFrame({"Feature_UAMS70": [0.4, 0.7]}, index=["P1", "P2"]).to_csv(score_path)
    monkeypatch.setenv("COMMPASSRISKSCOREFILE", str(score_path))
    training_features = pd.DataFrame(
        {"Feature_clin_D_PT_age": [60, 70]}, index=["P1", "P2"]
    )
    training_labels = pd.DataFrame(
        {"censpfs": [True, False], "pfscdy": [-2.0, 20.0]}, index=["P1", "P2"]
    )
    model_json = tmp_path / "GEP_UAMS70" / "pfs_shuffle0_fold0.json"
    results = {"endpoint": "pfs", "use_clin": True}

    model, _ = score_wiauc._build_coxph_model(
        results, model_json, training_features, training_labels
    )

    assert captured == {"feature_pattern": "Feature_GEP_UAMS70", "use_clin": True}
    assert model.training_data.columns.tolist() == [
        "Feature_clin_D_PT_age",
        "Feature_GEP_UAMS70",
    ]
    assert model.survival["time"].tolist() == [0.0, 22.0]


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