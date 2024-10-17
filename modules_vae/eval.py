import json
from glob import glob
import os
import numpy as np
# Set the working directory to ../ relative to the base directory of this file
base_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(base_dir, '..'))
from dotenv import load_dotenv
load_dotenv('.env')

def score_one_model(model_name,endpoint):
    
    jsonfiles=glob(f"{os.environ.get('OUTPUTDIR')}/{model_name}/{endpoint}_shuffle*_fold*.json")
    
    metrics = []
    
    try:
        for jsonfile in jsonfiles:
            with open(jsonfile, 'r') as f:
                result = json.load(f)
                metrics.append(result['best_epoch']['valid_metric'])
    except(KeyError):
        return
    
    if len(metrics)>0:
        average_metric = np.mean(metrics)
        std_metric = np.std(metrics)
        count = len(metrics)
        
        ci_lower = average_metric - 1.96 * (std_metric / np.sqrt(count))
        ci_upper = average_metric + 1.96 * (std_metric / np.sqrt(count))
        
        return {
            endpoint :
            {
                'mean': average_metric, 
                'CI lower': ci_lower, 
                'CI upper': ci_upper, 
                'N': count
            }
        }

def score_all_models(model_names,endpoint):
    scores = {model_name: score_one_model(model_name, endpoint) for model_name in model_names}
    return scores

if __name__ == "__main__":
    model_paths = glob(f"{os.environ.get('OUTPUTDIR')}/*")
    model_names = [os.path.basename(path) for path in model_paths]

    scores = score_all_models(model_names,'pfs')
    filtered_scores = {k: v for k, v in sorted(scores.items()) if v is not None}

    with open('model_scores.json', 'w') as f:
        json.dump(filtered_scores, f, indent=4)
