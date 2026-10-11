import argparse
import json
from pathlib import Path

import numpy as np


def summary_statistics_single_model(
    model_path: str | Path, endpoint: str | None = None
) -> dict:
    """
    @model_path: path to the model results folder
    the model folder will have 50 json files for OS and 50 for PFS
    in the form shuffleS_foldF.json, S=0-9, F=0-4
    """
    model_path = Path(model_path)
    model = {}
    endpoints = [endpoint] if endpoint else ['os', 'pfs']
    for current_endpoint in endpoints:
        model[current_endpoint] = {}
        jsonfiles = sorted(model_path.glob(f"{current_endpoint}_*.json"))
        metrics = {
            'valid_metric': [],
            'uams_metric': [],
            'hovon_metric': [],
            'emtab_metric': [],
            'apex_metric': [],
        }
        for jsonfile in jsonfiles:
            with jsonfile.open() as f:
                result = json.load(f)
            try:
                for metric_name in metrics:
                    metrics[metric_name].append(result['best_epoch'][metric_name])
            except KeyError:
                continue

        for metric_name, metric in metrics.items():
            count = len(metric)
            if count == 0:
                continue
            if count == 1:
                model[current_endpoint][metric_name] = metric[0]
                continue
            average_metric = np.mean(metric)
            std_metric = np.std(metric)
            ci_lower = average_metric - 1.96 * (std_metric / np.sqrt(count))
            ci_upper = average_metric + 1.96 * (std_metric / np.sqrt(count))
            model[current_endpoint][metric_name] = {
                'mean': average_metric, 
                'CI lower': ci_lower, 
                'CI upper': ci_upper, 
                'N': count
            }
    
    return model

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize OS and PFS validation metrics for one model folder."
    )
    parser.add_argument(
        "model_folder",
        type=Path,
        help="Folder containing the model's endpoint result JSON files",
    )
    parser.add_argument(
        "endpoint",
        nargs="?",
        choices=("os", "pfs"),
        help="Endpoint to summarize (default: summarize both OS and PFS)",
    )
    args = parser.parse_args()
    model_folder = args.model_folder.expanduser().resolve()
    if not model_folder.is_dir():
        parser.error(f"model_folder is not a directory: {model_folder}")

    endpoints = [args.endpoint] if args.endpoint else ["os", "pfs"]
    summary = summary_statistics_single_model(model_folder, args.endpoint)
    for endpoint in endpoints:
        output_path = model_folder / f"summary_{endpoint}.json"
        with output_path.open('w') as f:
            json.dump(summary[endpoint], f, indent=4)
            f.write("\n")
        print(f"Summary: {output_path}")


if __name__ == "__main__":
    main()
