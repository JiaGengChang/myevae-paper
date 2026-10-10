import argparse
import ast
import inspect
import json
import os
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


def _model_architecture(results: dict) -> str:
    architecture = results.get("params_fixed", {}).get("architecture")
    if architecture:
        return str(architecture).lower()
    if "endpoint" in results and "use_clin" in results:
        return "coxph"
    raise ValueError("Could not determine model architecture from the model JSON")


def _model_params(results: dict) -> dict:
    best_epoch = results.get("best_epoch", {})
    params = best_epoch.get("params") if isinstance(best_epoch, dict) else None
    if isinstance(params, dict) and params:
        return params
    params_fixed = results.get("params_fixed")
    if isinstance(params_fixed, dict) and params_fixed:
        return params_fixed
    raise ValueError("Model JSON does not contain model parameters")


def _model_input_types(architecture: str, params: dict, params_fixed: dict) -> tuple[list[str], list[str]]:
    if architecture == "vae":
        vae_inputs = _value(params["input_types"])
        subtask_inputs = _value(params["input_types_subtask"])
        if vae_inputs != ["exp"] or subtask_inputs != ["clin"]:
            raise ValueError("External parsers currently support VAE inputs ['exp'] and subtask inputs ['clin']")
        return vae_inputs, subtask_inputs

    input_types = _value(params.get("input_types_all", params_fixed.get("input_types_all")))
    if not isinstance(input_types, list) or not input_types:
        raise ValueError("Model JSON is missing a non-empty input_types_all list")
    unsupported = [input_type for input_type in input_types if input_type not in {"exp", "clin"}]
    if unsupported:
        raise ValueError(
            "External cohort parsers support only expression and clinical features; "
            f"this model requires {unsupported}"
        )
    return input_types, []


def _load_training_data(
    params_fixed: dict, feature_variant: str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_dir = Path(os.environ["SPLITDATADIR"])
    endpoint = params_fixed["endpoint"]
    feature_variant = feature_variant or "processed_mut_nan"
    if params_fixed.get("fulldata", False):
        if feature_variant is None:
            feature_variant = "processed" if params_fixed.get("architecture", "").lower() == "coxph" else "processed_mut_nan"
        features_path = split_dir / f"full_features_{endpoint}_{feature_variant}.parquet"
        labels_path = split_dir / "full_labels.parquet"
    else:
        split_dir = split_dir / str(params_fixed["shuffle"]) / str(params_fixed["fold"])
        features_path = split_dir / f"train_features_{endpoint}_{feature_variant}.parquet"
        labels_path = split_dir / "train_labels.parquet"
    missing = [path for path in (features_path, labels_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Training data file(s) not found: " + ", ".join(map(str, missing)))
    features = pd.read_parquet(features_path)
    labels = pd.read_parquet(labels_path)
    return features, labels


def _build_model(
    results: dict,
    train_features: pd.DataFrame,
    train_labels: pd.DataFrame,
    architecture: str,
    weights_file: Path,
):
    params_fixed = results["params_fixed"]
    params = _model_params(results)
    if params_fixed.get("endpoint") != "pfs":
        raise ValueError("External wiAUC scoring requires a PFS model")

    input_types, input_types_subtask = _model_input_types(architecture, params, params_fixed)

    event_col = params_fixed["eventcol"]
    duration_col = params_fixed["durationcol"]
    train_data = pd.concat([train_labels[[event_col, duration_col]], train_features], axis=1)

    if architecture == "vae":
        declared_input_dims = _value(params_fixed.get("input_dims"))
        if declared_input_dims is not None:
            from modules_vae.model import MultiModalVAE

            model = MultiModalVAE(
                input_types=input_types,
                input_dims=declared_input_dims,
                layer_dims=_value(params["layer_dims"]),
                input_types_subtask=input_types_subtask,
                input_dims_subtask=_value(params_fixed["input_dims_subtask"]),
                layer_dims_subtask=_value(params["layer_dims_subtask"]),
                z_dim=int(_value(params["z_dim"])),
                topKgenes=_value(params.get("topKgenes", params_fixed.get("topKgenes"))),
                activation=_activation(params.get("activation", "LeakyReLU()")),
                subtask_activation=_activation(params.get("subtask_activation", "Tanh()")),
            )
            model.load_state_dict(torch.load(weights_file, map_location="cpu", weights_only=True))
            model.eval()
            return model, input_types, input_types_subtask, "vae"

        estimator = VAE(
            input_types=input_types,
            subset_microarray=bool(_value(params.get("subset_microarray", params_fixed.get("subset", False)))),
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
            activation=_activation(params.get("activation", "LeakyReLU()")),
            subtask_activation=_activation(params.get("subtask_activation", "Tanh()")),
            scale_method=params["scale_method"],
            topKgenes=_value(params.get("topKgenes")),
        )
        estimator.fit(train_data)
        model = estimator.model
        model.load_state_dict(torch.load(weights_file, map_location="cpu", weights_only=True))
        model.eval()
        return model, input_types, input_types_subtask, "vae"

    subset_microarray = bool(_value(params_fixed.get("subset", False)))
    if architecture == "deepsurv":
        from modules_deepsurv.estimator import DeepSurv

        estimator = DeepSurv(
            input_types_all=input_types,
            subset_microarray=subset_microarray,
            layer_dims=_value(params["layer_dims"]),
            activation=_activation(params["activation"]),
            dropout=float(_value(params["dropout"])),
            lr=float(_value(params["lr"])),
            epochs=0,
            burn_in=int(_value(params["burn_in"])),
            patience=int(_value(params["patience"])),
            batch_size=int(_value(params["batch_size"])),
            eventcol=event_col,
            durationcol=duration_col,
            scale_method=params.get("scale_method"),
        )
        estimator.fit(train_data)
        model = estimator.model.net
        model.load_state_dict(torch.load(weights_file, map_location="cpu", weights_only=True))
        model.eval()
        return model, input_types, [], "torch"

    with weights_file.open() as stream:
        saved_estimator_params = json.load(stream)
    if architecture == "coxnet":
        from modules_coxnet.estimator import Coxnet

        estimator_type = Coxnet
    elif architecture == "rsf":
        from modules_rsf.estimator import RSF

        estimator_type = RSF
    else:
        raise ValueError(f"Unsupported model architecture: {architecture!r}")

    accepted_params = set(inspect.signature(estimator_type.__init__).parameters)
    estimator_params = {
        name: _value(value)
        for name, value in saved_estimator_params.items()
        if name in accepted_params
    }
    estimator_params.update({
        "eventcol": event_col,
        "durationcol": duration_col,
        "input_types_all": input_types,
        "subset_microarray": subset_microarray,
    })
    if "scale_method" in accepted_params:
        estimator_params["scale_method"] = params.get("scale_method", "std")
    estimator = estimator_type(**estimator_params)
    estimator.fit(train_data)
    return estimator.model, input_types, [], "sklearn"


def _predict_risk_scores(model, cohort: dict, input_types: list[str], prediction_type: str) -> np.ndarray:
    feature_tensors = [
        torch.as_tensor(cohort[input_type].to_numpy(), dtype=torch.float64)
        for input_type in input_types
    ]
    if prediction_type == "vae":
        expression = torch.as_tensor(cohort["exp"].to_numpy(), dtype=torch.float64)
        clinical = torch.as_tensor(cohort["clin"].to_numpy(), dtype=torch.float64)
        with torch.no_grad():
            _, _, _, risk_scores = model([[expression], [clinical]])
        return risk_scores.reshape(-1).cpu().numpy()

    features = torch.cat(feature_tensors, dim=-1)
    if prediction_type == "torch":
        with torch.no_grad():
            risk_scores = model(features)
        return risk_scores.reshape(-1).cpu().numpy()
    return np.asarray(model.predict(features.numpy()), dtype=float).reshape(-1)


def _external_coxph_inputs(endpoint: str) -> dict:
    from utils import parsers_external as parsers

    cohort_specs = [
        ("UAMS", "GSE24080UAMS", parsers.parse_clin_uams, "UAMSRISKSCOREFILE", "days"),
        ("HOVON65", "HOVON65", parsers.parse_clin_hovon, "HOVONRISKSCOREFILE", "days"),
        ("EMTAB4032", "EMTAB4032", parsers.parse_clin_emtab, "EMTABRISKSCOREFILE", "days"),
        ("APEX039", None, parsers.parse_clin_apex, "APEXRISKSCOREFILE", "months"),
    ]
    event_col = f"D_{endpoint.upper()}_FLAG"
    time_col = f"D_{endpoint.upper()}"
    cohort_data = {}
    for name, study, parse_clin, score_env, time_unit in cohort_specs:
        clinical = parse_clin()
        score_path = os.environ.get(score_env)
        if not score_path:
            raise ValueError(f"Environment variable {score_env} is required for CoxPH scoring")
        risk_scores = pd.read_csv(score_path, index_col=0)
        if not clinical.index.is_unique or not risk_scores.index.is_unique:
            raise ValueError(f"{name} has duplicate patient IDs in clinical or risk-score data")
        if study is None:
            survival = _apex_survival_frame(endpoint)
            outcome_event_col = f"{endpoint.upper()}_EVENT"
            outcome_time_col = endpoint.upper()
        else:
            survival = parsers.global_clinsurv_df.loc[
                parsers.global_clinsurv_df["Study"].eq(study), [event_col, time_col]
            ]
            outcome_event_col = event_col
            outcome_time_col = time_col
        survival = survival.reindex(clinical.index)
        if survival.isna().any().any():
            raise ValueError(f"{name} has missing survival data after patient alignment")
        cohort_data[name] = {
            "features": clinical.join(risk_scores.reindex(clinical.index)),
            "events": survival[outcome_event_col].to_numpy(dtype=bool),
            "times_days": survival[outcome_time_col].to_numpy(dtype=float)
            * (1.0 if time_unit == "days" else 365.25 / 12.0),
            "source_time_unit": time_unit,
        }
    return cohort_data


def _build_coxph_model(results: dict, model_json: Path, train_features: pd.DataFrame, train_labels: pd.DataFrame):
    from sksurv.util import Surv
    from utils.coxph import create_baseline_model

    model_name = model_json.parent.name
    use_clin = bool(results["use_clin"])
    if model_name.endswith("_full"):
        model_name = model_name.removesuffix("_full")
    if model_name.endswith("_noclin"):
        model_name = model_name.removesuffix("_noclin")
        use_clin = False
    if model_name not in {"Clin_only", "GEP_UAMS70", "GEP_SKY92", "GEP_IFM15", "GEP_MRC-IX-6"}:
        raise ValueError(f"Could not identify a supported CoxPH baseline from directory {model_json.parent.name!r}")

    score_path = os.environ.get("COMMPASSRISKSCOREFILE")
    if not score_path:
        raise ValueError("Environment variable COMMPASSRISKSCOREFILE is required for CoxPH scoring")
    training_clinical = train_features.filter(regex="Feature_clin")
    training_scores = pd.read_csv(score_path, index_col=0)
    training_data = training_clinical.join(training_scores)
    external = _external_coxph_inputs(results["endpoint"])
    external_columns = external["UAMS"]["features"].columns
    if len(training_data.columns) != len(external_columns):
        raise ValueError("Training clinical and risk-score columns do not match external CoxPH features")
    training_data.columns = external_columns

    event_col = f"cens{results['endpoint']}"
    duration_col = f"{results['endpoint']}cdy"
    events = train_labels[event_col].to_numpy(dtype=bool)
    times = train_labels[duration_col].to_numpy(dtype=float, copy=True)
    times -= min(0.0, float(np.min(times)))
    training_survival = Surv.from_arrays(events, times)
    model = create_baseline_model(f"Feature_{model_name}", use_clin)
    model.fit(training_data, training_survival)
    return model, external


def _apex_survival_frame(endpoint: str) -> pd.DataFrame:
    path = os.environ["APEXCLINDATAFILE"]
    frame = pd.read_csv(path, sep="\t", index_col=0).convert_dtypes()
    event_col = f"{endpoint.upper()}_EVENT"
    time_col = endpoint.upper()
    missing = frame[event_col].isna() | frame[time_col].isna()
    frame.loc[missing, event_col] = frame.loc[missing, "OS_EVENT"]
    frame.loc[missing, time_col] = frame.loc[missing, "OS"]
    return frame[[event_col, time_col]]


def _excluded_validation_cohorts(model_json: Path) -> set[str]:
    model_path = model_json.as_posix()
    excluded = set()
    if "UAMS" in model_path:
        excluded.add("UAMS")
    if "SKY92" in model_path:
        excluded.add("HOVON65")
    return excluded


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
    with model_json.open() as stream:
        results = json.load(stream)

    architecture = _model_architecture(results)
    if architecture == "coxph":
        if weights_path:
            raise ValueError("CoxPH baseline models are refit from training data and do not use a weights file")
        endpoint = results["endpoint"]
        if endpoint != "pfs":
            raise ValueError("External wiAUC scoring requires a PFS model")
        train_features, train_labels = _load_training_data(results, feature_variant="processed_mut_nan")
        model, external = _build_coxph_model(results, model_json, train_features, train_labels)
        cohorts = {
            name: {
                "events": cohort["events"],
                "times_days": cohort["times_days"],
                "risk_scores": np.asarray(model.predict(cohort["features"]), dtype=float).reshape(-1),
            }
            for name, cohort in external.items()
        }
        event_col = f"cens{endpoint}"
        duration_col = f"{endpoint}cdy"
        training_split = {
            "fulldata": bool(results.get("fulldata", False)),
            "shuffle": results.get("shuffle"),
            "fold": results.get("fold"),
        }
        weights_file = None
    else:
        params_fixed = results["params_fixed"]
        if params_fixed.get("endpoint") != "pfs":
            raise ValueError("External wiAUC scoring requires a PFS model")
        weights_file = _resolve_weights(model_json, weights_path)
        train_features, train_labels = _load_training_data(params_fixed)
        model, input_types, _, prediction_type = _build_model(
            results, train_features, train_labels, architecture, weights_file
        )
        params = _model_params(results)
        external = _external_inputs(params_fixed, params.get("scale_method", "std"))
        cohorts = {}
        for name, cohort in external.items():
            prediction_inputs = {
                "exp": cohort["expression"],
                "clin": cohort["clinical"],
            }
            cohorts[name] = {
                "events": cohort["events"],
                "times_days": cohort["times_days"],
                "risk_scores": _predict_risk_scores(
                    model, prediction_inputs, input_types, prediction_type
                ),
            }
        event_col = params_fixed["eventcol"]
        duration_col = params_fixed["durationcol"]
        training_split = {
            "fulldata": bool(params_fixed.get("fulldata", False)),
            "shuffle": params_fixed.get("shuffle"),
            "fold": params_fixed.get("fold"),
        }

    report = calculate_wiauc(
        train_labels[event_col].to_numpy(dtype=bool),
        train_labels[duration_col].to_numpy(dtype=float),
        cohorts,
        excluded_cohorts=_excluded_validation_cohorts(model_json),
    )
    report["model_json"] = str(model_json)
    report["architecture"] = architecture
    report["weights_file"] = str(weights_file.resolve()) if weights_file else None
    report["training_split"] = training_split
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