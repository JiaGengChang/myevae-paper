import torch
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import argparse
from torch.utils.data import DataLoader
from sksurv.metrics import concordance_index_censored
import json
import sys
from dotenv import load_dotenv
load_dotenv("../.env")
sys.path.append(os.environ.get("PROJECTDIR"))
from modules_vae.params import VAEParams as no_argument_params
from modules_vae.model import MultiModalVAE as Model
from utils.parsers_external import *
from utils.scaler_external import scale_and_impute_external_dataset as scale_impute

def score_external_datasets(model,endpoint,shuffle,fold,level="affy"):
    params = no_argument_params()
    
    genes = helper_get_training_genes(endpoint,shuffle,fold)

    uams_clin_tensor = torch.tensor(scale_impute(parse_clin_uams(), params.scale_method).values)
    uams_exp_tensor = torch.tensor(scale_impute(parse_exp_uams(genes,level), params.scale_method).values)

    hovon_clin_tensor = torch.tensor(scale_impute(parse_clin_hovon(), params.scale_method).values)
    hovon_exp_tensor = torch.tensor(scale_impute(parse_exp_hovon(genes,level), params.scale_method).values)

    emtab_clin_tensor = torch.tensor(scale_impute(parse_clin_emtab(), params.scale_method).values)
    emtab_exp_tensor = torch.tensor(scale_impute(parse_exp_emtab(genes,level), params.scale_method).values)
    
    uams_events, uams_times = parse_surv_uams(endpoint)
    hovon_events, hovon_times = parse_surv_hovon(endpoint)
    emtab_events, emtab_times = parse_surv_emtab(endpoint)
    
    model.eval()           
    with torch.no_grad():
        _, _, _, estimates_uams = model([[uams_exp_tensor], [uams_clin_tensor]])
        _, _, _, estimates_hovon = model([[hovon_exp_tensor], [hovon_clin_tensor]])
        _, _, _, estimates_emtab = model([[emtab_exp_tensor], [emtab_clin_tensor]])

    cindex_uams = concordance_index_censored(uams_events, uams_times, estimates_uams.flatten())[0]
    cindex_hovon = concordance_index_censored(hovon_events, hovon_times, estimates_hovon.flatten())[0]
    cindex_emtab = concordance_index_censored(emtab_events, emtab_times, estimates_emtab.flatten())[0]
        
    return max(cindex_uams,1-cindex_uams), max(cindex_hovon,1-cindex_hovon), max(cindex_emtab,1-cindex_emtab)    

if __name__=="__main__":
    """
    Parse the 3 arguments which we will parallelize across. 
    the actual hyperparameters to modify are in params.py
    """
    parser = argparse.ArgumentParser(description='Score an VAE model on external datasets then appends scores to json results.')
    parser.add_argument('endpoint', type=str, choices=['pfs', 'os'], default='pfs', help='Survival endpoint (pfs or os)')
    parser.add_argument('shuffle', type=int, choices=range(10), default=0, help='Random state for k-fold splitting (0-9)')
    parser.add_argument('fold', type=int, choices=range(5), default=0, help='Fold index for k-fold splitting (0-4)')

    args = parser.parse_args()
    params = specify_params_here(args.endpoint)

    model = Model(params.input_types,
                params.input_dims,
                params.layer_dims,
                params.input_types_subtask,
                params.input_dims_subtask,
                params.layer_dims_subtask,
                params.z_dim)

    with open(f"{params.resultsprefix}.json",'r') as f:
        results = json.load(f)
    
    model.load_state_dict(torch.load(f"{params.resultsprefix}.pth",weights_only=True))
    
    cindex_uams, cindex_hovon, cindex_emtab = score_external_datasets(model,args.endpoint)
    results['best_epoch']['uams_metric'] = cindex_uams
    results['best_epoch']['hovon_metric'] = cindex_hovon
    results['best_epoch']['emtab_metric'] = cindex_emtab

    with open(f"{params.resultsprefix}.json", 'w') as f:
        json.dump(results, f)