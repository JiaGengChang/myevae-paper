import argparse
import json
from pathlib import Path

import numpy as np


def _summary_statistics(scores: list[float]) -> dict[str, float | int]:
    count = len(scores)
    mean_score = float(np.mean(scores))
    ci_margin = 1.96 * (float(np.std(scores)) / np.sqrt(count))
    return {
        "mean": mean_score,
        "CI lower": mean_score - ci_margin,
        "CI upper": mean_score + ci_margin,
        "N": count,
    }


def _cohort_name(name: str) -> str:
    """Use familiar dataset names while retaining unknown cohort labels."""
    aliases = {
        "EMTAB4032": "EMTAB",
        "HOVON65": "GMMG-HD4",
        "APEX039": "APEX",
    }
    return aliases.get(name.upper(), name)


def summary_statistics_wiauc(
    model_path: str | Path,
) -> dict[str, float | int | dict[str, dict[str, float | int]]]:
    """Summarize wiAUC and per-cohort iAUC values in a model results folder."""
    model_path = Path(model_path)
    json_files = sorted(model_path.glob("*_wiauc.json"))
    if not json_files:
        raise FileNotFoundError(f"No *_wiauc.json files found in {model_path}")

    scores = []
    cohort_scores: dict[str, list[float]] = {}
    for json_file in json_files:
        with json_file.open() as stream:
            result = json.load(stream)
        if not isinstance(result, dict) or "wiAUC" not in result:
            raise ValueError(f"Missing wiAUC score in {json_file}")

        score = result["wiAUC"]
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError(f"wiAUC score must be numeric in {json_file}")
        if not np.isfinite(score):
            raise ValueError(f"wiAUC score must be finite in {json_file}")
        scores.append(float(score))

        cohorts = result.get("cohorts")
        if cohorts is None:
            continue
        if not isinstance(cohorts, dict):
            raise ValueError(f"cohorts must be an object in {json_file}")
        for name, cohort in cohorts.items():
            if not isinstance(cohort, dict) or "iAUC" not in cohort:
                raise ValueError(f"Missing iAUC for cohort {name!r} in {json_file}")
            iauc = cohort["iAUC"]
            if isinstance(iauc, bool) or not isinstance(iauc, (int, float)):
                raise ValueError(f"iAUC for cohort {name!r} must be numeric in {json_file}")
            if not np.isfinite(iauc):
                raise ValueError(f"iAUC for cohort {name!r} must be finite in {json_file}")
            cohort_scores.setdefault(_cohort_name(name), []).append(float(iauc))

    summary = _summary_statistics(scores)
    summary["cohorts"] = {
        name: _summary_statistics(values) for name, values in sorted(cohort_scores.items())
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize wiAUC scores across shuffle/fold JSON reports."
    )
    parser.add_argument(
        "model_path",
        type=Path,
        help="Folder containing *_wiauc.json files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output summary JSON path (default: <model_path>/wiauc_summary.json)",
    )
    args = parser.parse_args()

    if not args.model_path.is_dir():
        parser.error(f"model_path is not a directory: {args.model_path}")

    summary = summary_statistics_wiauc(args.model_path)
    output_path = args.output or args.model_path / "wiauc_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as stream:
        json.dump(summary, stream, indent=4)
        stream.write("\n")
    print(f"Summary: {output_path}")
    print(
        f"wiAUC: mean={summary['mean']:.6f}, "
        f"95% CI=({summary['CI lower']:.6f}, {summary['CI upper']:.6f}), "
        f"N={summary['N']}"
    )


if __name__ == "__main__":
    main()
