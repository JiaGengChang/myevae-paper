import json
from glob import glob
import os
import numpy as np
# Set the working directory to ../ relative to the base directory of this file
base_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(base_dir, '..'))
from dotenv import load_dotenv
import argparse
load_dotenv('.env')

def summary_statistics_single_model(model_name):
    model = {}
    for endpoint in ['os', 'pfs']:
        model[endpoint] = {}
        jsonfiles=glob(f"{os.environ.get('OUTPUTDIR')}/{model_name}/{endpoint}_shuffle*_fold*.json")    
        valid_metric = []
        uams_metric = []
        hovon_metric = []
        emtab_metric = []
        for jsonfile in jsonfiles:
            with open(jsonfile, 'r') as f:
                result = json.load(f)
            try:
                valid_metric.append(result['best_epoch']['valid_metric'])
                uams_metric.append(result['best_epoch']['uams_metric'])
                hovon_metric.append(result['best_epoch']['hovon_metric'])
                emtab_metric.append(result['best_epoch']['emtab_metric'])
            except(KeyError):
                continue
        
        for metric_name in ['valid_metric', 'uams_metric', 'hovon_metric', 'emtab_metric']:
            metric = eval(metric_name)
            count = len(metric)
            if count == 0:
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

def summary_statistics_many_models(model_names):
    scores = {model_name: summary_statistics_single_model(model_name) for model_name in model_names}
    return scores

if __name__ == "__main__":
    model_paths = glob(f"{os.environ.get('OUTPUTDIR')}/*")
    model_names = [os.path.basename(path) for path in model_paths]

    scores = summary_statistics_many_models(model_names)
    filtered_scores = {k: v for k, v in sorted(scores.items()) if v is not None}

    with open('model_scores.json', 'w') as f:
        json.dump(filtered_scores, f, indent=4)
