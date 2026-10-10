import argparse
import ast
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from modules_vae.estimator import VAE
from utils.scaler_external import scale_and_impute_external_dataset
from utils.wiauc import calculate_wiauc


def _value(value):
    if not isinstance(value, str):
        return value
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def _activation(value):
    expression = ast.parse(value, mode="eval").body
    if not isinstance(expression, ast.Call):
        raise ValueError(f"Unsupported activation expression: {value}")
    if isinstance(expression.func, ast.Name):
        name = expression.func.id
    elif isinstance(expression.func, ast.Attribute):
        name = expression.func.attr
    else:
        raise ValueError(f"Unsupported activation expression: {value}")
    activation_type = getattr(torch.nn, name, None)
    if activation_type is None or not issubclass(activation_type, torch.nn.Module):
        raise ValueError(f"Unsupported PyTorch activation: {name}")
    args = [ast.literal_eval(argument) for argument in expression.args]
    kwargs = {
        keyword.arg: ast.literal_eval(keyword.value)
        for keyword in expression.keywords
        if keyword.arg is not None
    }
    if len(kwargs) != len(expression.keywords):
        raise ValueError(f"Unsupported activation arguments: {value}")
    return activation_type(*args, **kwargs)


def _resolve_weights(model_json: Path, weights_path: str | None) -> Path:
    if weights_path:
        path = Path(weights_path).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if path.is_file():
            return path
        raise FileNotFoundError(f"Weights file not found: {path}")

    candidates = [model_json.with_suffix(".pth"), Path(f"{model_json}.pth")]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "Could not find model weights; checked " + " and ".join(map(str, candidates))
    )


def _load_training_data(params_fixed: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_dir = Path(os.environ["SPLITDATADIR"])
    endpoint = params_fixed["endpoint"]
    if params_fixed.get("fulldata", False):
        features_path = split_dir / f"full_features_{endpoint}_processed_mut_nan.parquet"
        labels_path = split_dir / "full_labels.parquet"
    else:
        split_dir = split_dir / str(params_fixed["shuffle"]) / str(params_fixed["fold"])
        features_path = split_dir / f"train_features_{endpoint}_processed_mut_nan.parquet"
        labels_path = split_dir / "train_labels.parquet"
    missing = [path for path in (features_path, labels_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Training data file(s) not found: " + ", ".join(map(str, missing)))
    features = pd.read_parquet(features_path)
    labels = pd.read_parquet(labels_path)
    return features, labels


def _build_model(results: dict, train_features: pd.DataFrame, train_labels: pd.DataFrame):
    params_fixed = results["params_fixed"]
    params = results["best_epoch"]["params"]
    architecture = params_fixed.get("architecture", "").lower()
    if architecture != "vae":
        raise ValueError(f"This scorer currently supports VAE checkpoints, not {architecture!r}")
    if params_fixed.get("endpoint") != "pfs":
        raise ValueError("External wiAUC scoring requires a PFS model")

    input_types = _value(params["input_types"])
    input_types_subtask = _value(params["input_types_subtask"])
    if input_types != ["exp"] or input_types_subtask != ["clin"]:
        raise ValueError("External parsers currently support VAE inputs ['exp'] and subtask inputs ['clin']")

    event_col = params_fixed["eventcol"]
    duration_col = params_fixed["durationcol"]
    train_data = pd.concat([train_labels[[event_col, duration_col]], train_features], axis=1)
    estimator = VAE(
        input_types=input_types,
        subset_microarray=bool(_value(params.get("subset_microarray", "False"))),
        layer_dims=_value(params["layer_dims"]),
        input_types_subtask=input_types_subtask,
        layer_dims_subtask=_value(params["layer_dims_subtask"]),
        z_dim=int(_value(params["z_dim"])),
        lr=float(_value(params["lr"])),
        batch_size=int(_value(params["batch_size"])),
        epochs=0,
        burn_in=int(_value(params["burn_in"])),
        patience=int(_value(params["patience"])),
        eventcol=event_col,
        durationcol=duration_col,
        kl_weight=float(_value(params["kl_weight"])),
        activation=_activation(params["activation"]),
        subtask_activation=_activation(params["subtask_activation"]),
        scale_method=params["scale_method"],
        topKgenes=_value(params.get("topKgenes")),
    )
    estimator.fit(train_data)
    return estimator.model


def _apex_survival_frame(endpoint: str) -> pd.DataFrame:
    path = os.environ["APEXCLINDATAFILE"]
    frame = pd.read_csv(path, sep="\t", index_col=0).convert_dtypes()
    event_col = f"{endpoint.upper()}_EVENT"
    time_col = endpoint.upper()
    missing = frame[event_col].isna() | frame[time_col].isna()
    frame.loc[missing, event_col] = frame.loc[missing, "OS_EVENT"]
    frame.loc[missing, time_col] = frame.loc[missing, "OS"]
    return frame[[event_col, time_col]]


def _external_inputs(params_fixed: dict, scale_method: str):
    from utils import parsers_external as parsers

    endpoint = params_fixed["endpoint"].upper()
    event_col = f"D_{endpoint}_FLAG"
    time_col = f"D_{endpoint}"
    genes = params_fixed["genes"]
    cohort_specs = [
        ("UAMS", "GSE24080UAMS", parsers.parse_clin_uams, parsers.parse_exp_uams, "days"),
        ("HOVON65", "HOVON65", parsers.parse_clin_hovon, parsers.parse_exp_hovon, "days"),
        ("EMTAB4032", "EMTAB4032", parsers.parse_clin_emtab, parsers.parse_exp_emtab, "days"),
        ("APEX039", None, parsers.parse_clin_apex, parsers.parse_exp_apex, "months"),
    ]
    cohort_data = {}
    for name, study, parse_clin, parse_exp, time_unit in cohort_specs:
        clinical = parse_clin()
        expression = parse_exp(genes, "affy")
        if not clinical.index.is_unique or not expression.index.is_unique:
            raise ValueError(f"{name} has duplicate patient IDs in clinical or expression data")
        missing_expression = clinical.index.difference(expression.index)
        if len(missing_expression):
            raise ValueError(f"{name} is missing expression data for {len(missing_expression)} patients")
        expression = expression.reindex(clinical.index)

        if study is None:
            survival = _apex_survival_frame(params_fixed["endpoint"])
        else:
            survival = parsers.global_clinsurv_df.loc[
                parsers.global_clinsurv_df["Study"].eq(study), [event_col, time_col]
            ]
        missing_survival = clinical.index.difference(survival.index)
        if len(missing_survival):
            raise ValueError(f"{name} is missing PFS data for {len(missing_survival)} patients")
        survival = survival.reindex(clinical.index)
        if survival.isna().any().any():
            raise ValueError(f"{name} has missing PFS events or times after cohort alignment")

        clinical_scaled = scale_and_impute_external_dataset(clinical, scale_method)
        expression_scaled = scale_and_impute_external_dataset(expression, scale_method)
        cohort_data[name] = {
            "clinical": clinical_scaled,
            "expression": expression_scaled,
            "events": survival[event_col if study is None else event_col].to_numpy(dtype=bool)
            if study is not None
            else survival[f"{params_fixed['endpoint'].upper()}_EVENT"].to_numpy(dtype=bool),
            "times_days": survival[time_col if study is not None else params_fixed["endpoint"].upper()].to_numpy(dtype=float)
            * (1.0 if time_unit == "days" else 365.25 / 12.0),
            "source_time_unit": time_unit,
        }
    return cohort_data


def score_model(model_json: Path, weights_path: str | None = None, report_path: str | None = None) -> Path:
    if not model_json.is_absolute():
        model_json = PROJECT_ROOT / model_json
    model_json = model_json.resolve()
    if not model_json.is_file():
        raise FileNotFoundError(f"Model JSON not found: {model_json}")
    weights_file = _resolve_weights(model_json, weights_path)
    with model_json.open() as stream:
        results = json.load(stream)

    train_features, train_labels = _load_training_data(results["params_fixed"])
    model = _build_model(results, train_features, train_labels)
    state_dict = torch.load(weights_file, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    params = results["best_epoch"]["params"]
    external = _external_inputs(results["params_fixed"], params["scale_method"])
    cohorts = {}
    with torch.no_grad():
        for name, cohort in external.items():
            dtype = next(model.parameters()).dtype
            expression = torch.as_tensor(cohort["expression"].to_numpy(), dtype=dtype)
            clinical = torch.as_tensor(cohort["clinical"].to_numpy(), dtype=dtype)
            _, _, _, risk_scores = model([[expression], [clinical]])
            cohorts[name] = {
                "events": cohort["events"],
                "times_days": cohort["times_days"],
                "risk_scores": risk_scores.reshape(-1).cpu().numpy(),
            }

    labels = train_labels
    report = calculate_wiauc(
        labels[results["params_fixed"]["eventcol"]].to_numpy(dtype=bool),
        labels[results["params_fixed"]["durationcol"]].to_numpy(dtype=float),
        cohorts,
    )
    report["model_json"] = str(model_json)
    report["weights_file"] = str(weights_file.resolve())
    report["training_split"] = {
        "shuffle": results["params_fixed"]["shuffle"],
        "fold": results["params_fixed"]["fold"],
    }
    report["cohort_time_units"] = {
        name: cohort["source_time_unit"] for name, cohort in external.items()
    }

    if report_path is None:
        output_path = model_json.with_name(f"{model_json.stem}_wiauc.json")
    else:
        output_path = Path(report_path).expanduser()
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a trained PFS VAE on external cohorts with wiAUC")
    parser.add_argument("model_json", help="Path to the trained model JSON")
    parser.add_argument("--weights", help="Optional path to the trained .pth checkpoint")
    parser.add_argument("--output", help="Optional path for the wiAUC JSON report")
    args = parser.parse_args()
    output_path = score_model(Path(args.model_json), args.weights, args.output)
    with output_path.open() as stream:
        report = json.load(stream)
    print(f"Report: {output_path}")
    for name, values in report["cohorts"].items():
        print(f"{name}: iAUC={values['iAUC']:.6f}, n_high_risk={values['n_high_risk']}")
    print(f"wiAUC: {report['wiAUC']:.6f}")


if __name__ == "__main__":
    main()