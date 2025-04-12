import json
from argparse import ArgumentParser
from torch import load as torch_load
from torch.nn import * # for activations
from datetime import datetime

import os
import sys
from dotenv import load_dotenv
assert load_dotenv("../.env") or load_dotenv('.env')
sys.path.append(os.environ.get("PROJECTDIR"))
from utils.parsers_external import *
from utils.validation import score_external_datasets
from utils.attrdict import AttrDict
from modules_deepsurv.estimator import DeepSurv
from modules_vae.estimator import VAE

def main(architecture:str,
         model_name:str,
         endpoint:str,
         shuffle:int,
         fold:int,
         fulldata:bool,
         subset:bool,
         resultspath:str=None,
         statedictpath:str=None) -> None:
    # FIXME: VAE is not implemented
    if architecture=="VAE":
        raise NotImplementedError
    
    # assumes model is trained on full data by default
    if resultspath is None:
        if fulldata:
            if subset:
                resultspath = f"{os.environ.get("OUTPUTDIR")}/{architecture.lower()}_models/{model_name}_subset_full/{endpoint}_full.json"
            else:
                resultspath = f"{os.environ.get("OUTPUTDIR")}/{architecture.lower()}_models/{model_name}_full/{endpoint}_full.json"
        else:
            resultspath = f"{os.environ.get("OUTPUTDIR")}/{architecture.lower()}_models/{model_name}/{endpoint}_shuffle{shuffle}_fold{fold}.json"
    if statedictpath is None:
        if fulldata:
            if subset:
                statedictpath = f"{os.environ.get("OUTPUTDIR")}/{architecture.lower()}_models/{model_name}_subset_full/{endpoint}_full.pth"
            else:
                statedictpath = f"{os.environ.get("OUTPUTDIR")}/{architecture.lower()}_models/{model_name}_full/{endpoint}_full.pth"
        else:
            statedictpath = f"{os.environ.get("OUTPUTDIR")}/{architecture.lower()}_models/{model_name}/{endpoint}_shuffle{shuffle}_fold{fold}.pth"        
    assert os.path.exists(resultspath)
    assert os.path.exists(statedictpath)
    
    results = json.load(open(resultspath,'r'))
    genes = results['params_fixed']['genes']
    params = results['best_epoch']['params'] # hyperparameters for fit
    # params is a dictionary
    params["genes"] = genes # genes seen by the model
    params["endpoint"] = endpoint # accessed by score_external_datasets

    best_estimator = \
    DeepSurv(input_types_all=eval(params['input_types_all']),
            subset_microarray=subset,
            layer_dims=eval(params['layer_dims']),
            activation=eval(params['activation']),
            dropout=eval(params['dropout']),
            lr=eval(params['lr']),
            epochs=0, # fit for 0 epochs
            burn_in=eval(params['burn_in']),
            patience=eval(params['patience']),
            batch_size=eval(params['batch_size']),
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
    state_dict = torch_load(statedictpath)
    best_estimator.model.net.load_state_dict(state_dict)
    model = best_estimator.model.net

    params_attr = AttrDict()
    params_attr.genes = params['genes']
    params_attr.endpoint = params['endpoint']
    params_attr.architecture = architecture
    params_attr.scale_method = params['scale_method']
    params_attr.input_types_all = eval(params['input_types_all'])

    cindex_uams, cindex_hovon, cindex_emtab, cindex_apex = score_external_datasets(model, params_attr)
    results['best_epoch']['uams_metric'] = cindex_uams
    results['best_epoch']['hovon_metric'] = cindex_hovon
    results['best_epoch']['emtab_metric'] = cindex_emtab
    results['best_epoch']['apex_metric'] = cindex_apex # APEX has no clinical data at all
    results['best_epoch']['timestamp'] = datetime.now().__str__() # add timestamp
    
    # update values in json results file
    with  open(resultspath,'w') as f:
        json.dump(results, f, indent=4)

    # DONE


if __name__ == "__main__":
    parser = ArgumentParser(description='Scoring of trained model')
    parser.add_argument('-a','--architecture',type=str,choices=['VAE','Deepsurv'],default='VAE')
    parser.add_argument('-m','--model_name',type=str,default='exp')
    parser.add_argument('-e','--endpoint',choices=['both','os','pfs'],default='both')
    parser.add_argument('-f','--fulldata',action='store_true',help='Was model trained on full CoMMpass data')
    parser.add_argument('-s','--subset',action='store_true',help='Was model trained on microarray subset genes')
    parser.add_argument('-p','--paramsfile',type=str,default=None,help='Path to the .json params file')
    parser.add_argument('-w','--weightsfile',type=str,default=None,help='Path to the .pth weights file')

    args = parser.parse_args()
    _pbs_array_id = int(os.getenv('PBS_ARRAY_INDEX', "-1"))
    pbs_shuffle=_pbs_array_id%10
    pbs_fold=_pbs_array_id//10
    
    if args.endpoint=="both":
        main(args.architecture, args.model_name, 'os', pbs_shuffle, pbs_fold, args.fulldata, args.subset, args.paramsfile, args.weightsfile)
        main(args.architecture, args.model_name, 'pfs', pbs_shuffle, pbs_fold, args.fulldata, args.subset, args.paramsfile, args.weightsfile)
    else:
        main(args.architecture, args.model_name, args.endpoint, pbs_shuffle, pbs_fold, args.fulldata, args.subset, args.paramsfile, args.weightsfile)