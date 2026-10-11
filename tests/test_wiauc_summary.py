import json
from importlib import import_module

import pytest


wiauc_summary = import_module("pipeline.5_wiauc_summary")


def test_summary_statistics_wiauc_aggregates_reports(tmp_path):
    reports = (
        (
            0.5,
            {
                "APEX039": 0.9,
                "EMTAB4032": 0.7,
                "UAMS": 0.6,
                "CoMMpass": 0.55,
                "HOVON65": 0.5,
            },
        ),
        (
            0.7,
            {
                "UAMS": 0.8,
                "HOVON65": 0.5,
                "CoMMpass": 0.65,
                "APEX039": 0.7,
                "EMTAB4032": 0.6,
            },
        ),
        (
            0.9,
            {
                "Masaryk": 0.8,
                "APEX039": 0.9,
                "UAMS": 1.0,
                "CoMMpass": 0.95,
                "HOVON65": 0.85,
            },
        ),
    )
    for index, (score, cohort_scores) in enumerate(reports):
        (tmp_path / f"pfs_shuffle{index}_fold0_wiauc.json").write_text(
            json.dumps(
                {
                    "wiAUC": score,
                    "cohorts": {
                        name: {
                            "iAUC": iauc,
                            "n_patients": 100 + index,
                            "cohort_weight": 1.0,
                            "included_in_wiauc": name != "UAMS",
                        }
                        for name, iauc in cohort_scores.items()
                    },
                }
            )
        )
    (tmp_path / "not_a_score.json").write_text(json.dumps({"wiAUC": 0.0}))

    result = wiauc_summary.summary_statistics_wiauc(tmp_path)

    expected_margin = 1.96 * (0.08**0.5 / 3)
    assert result["mean"] == pytest.approx(0.7)
    assert result["CI lower"] == pytest.approx(0.7 - expected_margin)
    assert result["CI upper"] == pytest.approx(0.7 + expected_margin)
    assert result["N"] == 3
    assert result["cohorts"]["UAMS"]["mean"] == pytest.approx(0.8)
    assert result["cohorts"]["UAMS"]["N"] == 3
    assert result["cohorts"]["Masaryk"]["mean"] == pytest.approx(0.7)
    assert result["cohorts"]["HOVON65"]["mean"] == pytest.approx((0.5 + 0.5 + 0.85) / 3)
    assert result["cohorts"]["APEX039"]["mean"] == pytest.approx((0.9 + 0.7 + 0.9) / 3)
    assert result["cohorts"]["CoMMpass"]["mean"] == pytest.approx((0.55 + 0.65 + 0.95) / 3)
    assert result["cohorts"]["CoMMpass"]["N"] == 3
    assert list(result["cohorts"]) == [
        "CoMMpass",
        "UAMS",
        "HOVON65",
        "Masaryk",
        "APEX039",
    ]


def test_summary_statistics_wiauc_requires_matching_json_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="No \\*_wiauc.json files"):
        wiauc_summary.summary_statistics_wiauc(tmp_path)


def test_summary_statistics_wiauc_rejects_report_without_score(tmp_path):
    (tmp_path / "pfs_shuffle0_fold0_wiauc.json").write_text(json.dumps({}))

    with pytest.raises(ValueError, match="Missing wiAUC score"):
        wiauc_summary.summary_statistics_wiauc(tmp_path)
