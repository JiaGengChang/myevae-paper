import json
from glob import glob
import os
import numpy as np
from dotenv import load_dotenv
assert load_dotenv('.env') or load_dotenv('../.env')

def summary_statistics_single_model(model_path):
    """
    @model_path: full path to the model results folder
    the model folder will have 50 json files for OS and 50 for PFS
    in the form shuffleS_foldF.json, S=0-9, F=0-4
    """
    model = {}
    # output directory folder is VAE_models or deepsurv_models
    for endpoint in ['os', 'pfs']:
        model[endpoint] = {}
        jsonfiles=glob(f"{model_path}/{endpoint}_*.json") # shuffle*_fold* or full 
        valid_metric = []
        uams_metric = []
        hovon_metric = []
        emtab_metric = []
        apex_metric = []
        for jsonfile in jsonfiles:
            with open(jsonfile, 'r') as f:
                result = json.load(f)
            try:
                valid_metric.append(result['best_epoch']['valid_metric'])
                uams_metric.append(result['best_epoch']['uams_metric'])
                hovon_metric.append(result['best_epoch']['hovon_metric'])
                emtab_metric.append(result['best_epoch']['emtab_metric'])
                apex_metric.append(result['best_epoch']['apex_metric'])
            except(KeyError):
                continue
        
        for metric_name in ['valid_metric', 'uams_metric', 'hovon_metric', 'emtab_metric', 'apex_metric']:
            metric = eval(metric_name)
            count = len(metric)
            if count == 0:
                continue
            if count == 1:
                model[endpoint][metric_name] = metric[0]
                continue
            average_metric = np.mean(metric)
            std_metric = np.std(metric)
            ci_lower = average_metric - 1.96 * (std_metric / np.sqrt(count))
            ci_upper = average_metric + 1.96 * (std_metric / np.sqrt(count))
            model[endpoint][metric_name] = {
                'mean': average_metric, 
                'CI lower': ci_lower, 
                'CI upper': ci_upper, 
                'N': count
            }
    
    return model

if __name__ == "__main__":
    model_paths = glob(f"{os.environ.get('OUTPUTDIR')}/*_models/*")

    scores = {model_path: summary_statistics_single_model(model_path) for model_path in model_paths}
    filtered_scores = {k: v for k, v in sorted(scores.items()) if v is not None}

    with open(f'{os.environ.get("PROJECTDIR")}/model_scores.json', 'w') as f:
        json.dump(filtered_scores, f, indent=4)
