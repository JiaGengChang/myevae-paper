import json
from argparse import ArgumentParser
from torch import load as torch_load
from torch.nn import * # for activations
from utils.validation import score_external_datasets

import os
import sys
from dotenv import load_dotenv
load_dotenv("../.env")
sys.path.append(os.environ.get("PROJECTDIR"))
from utils.parsers_external import *
from modules_deepsurv.estimator import DeepSurv
from modules_vae.estimator import VAE

def main(arch,model_name,endpoint,resultspath=None,statedictpath=None):
    if arch=="VAE":
        raise NotImplementedError
    
    # assumes model is trained on full data by default
    if resultspath is None:
        resultspath = f"{os.environ.get("OUTPUTDIR")}/{arch.lower()}_models/{model_name}_full/{endpoint}_full.json"
    if statedictpath is None:
        statedictpath = f"{os.environ.get("OUTPUTDIR")}/{arch.lower()}_models/{model_name}_full/{endpoint}_full.pth"
    assert os.path.exists(resultspath)
    assert os.path.exists(statedictpath)
    state_dict = torch_load(statedictpath)
    results = json.load(open(resultspath,'r'))
    params = results['best_epoch']['params'] # hyperparameters for fit
    setattr(params, "genes", results['params_fixed']['genes']) # genes seen by the model
    setattr(params, "endpoint", endpoint) # accessed by score_external_datasets

    best_estimator = DeepSurv(input_types_all=eval(params['input_types_all']),
                    subset_microarray=eval(params['subset_microarray']),
                    layer_dims=eval(params['layer_dims']),
                    activation=eval(params['activation']), # requires torch.nn.*
                    dropout=0, lr=0, epochs=0,
                    burn_in=0,
                    patience=0,
                    batch_size=1024,
                    eventcol=results['params_fixed']['eventcol'],
                    durationcol=results['params_fixed']['durationcol'],
                    scale_method=params['scale_method'])
    
    # get sample data to fit on
    splitsdir=os.environ.get("SPLITDATADIR")
    train_features_file=f'{splitsdir}/full_features_{endpoint}_processed.parquet'
    train_labels_file=f'{splitsdir}/full_labels.parquet'
    assert os.path.exists(train_features_file) and os.path.exists(train_labels_file)
    train_features=pd.read_parquet(train_features_file)
    train_labels=pd.read_parquet(train_labels_file)[[results['params_fixed']['eventcol'],results['params_fixed']['durationcol']]]
    train_dataframe=pd.concat([train_labels,train_features],axis=1)
    
    # call fit to establish torch module inside the class
    best_estimator.fit(train_dataframe)
    best_estimator.model.net.load_state_dict(state_dict)
    model = best_estimator.model.net

    cindex_uams, cindex_hovon, cindex_emtab, cindex_apex = score_external_datasets(best_estimator,params)
    results['best_epoch']['uams_metric'] = cindex_uams
    results['best_epoch']['hovon_metric'] = cindex_hovon
    results['best_epoch']['emtab_metric'] = cindex_emtab
    results['best_epoch']['apex_metric'] = cindex_apex # APEX has no clinical data at all

    # update values in json results file
    with  open('test.json','r') as f:
        json.dump(results, f)

    # DONE


if __name__ == "__main__":
    ArgumentParser.add_argument('-a','--architecture',type=str,choices=['VAE','Deepsurv'])
    ArgumentParser.add_argument('-m','--model_name',type=str)
    ArgumentParser.add_argument('-p','--paramsfile',type=str,default=None,help='Path to the .json params file')
    ArgumentParser.add_argument('-w','--weightsfile',type=str,default=None,help='Path to the .pth weights file')
    ArgumentParser.add_argument('-e','--endpoint',default='both',choices=['both','os','pfs'])

    args = ArgumentParser.parse_args()
    
    if args.endpoint=="both":
        main(args.architecture, args.model_name, 'os', args.paramsfile, args.weightsfile)
        main(args.architecture, args.model_name, 'pfs', args.paramsfile, args.weightsfile)
    else:
        main(args.architecture, args.model_name, args.endpoint, args.paramsfile, args.weightsfile)