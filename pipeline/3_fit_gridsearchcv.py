import os
from argparse import ArgumentParser
from json import dump as json_dump
import pandas as pd
from sklearn.model_selection import GridSearchCV

from dotenv import load_dotenv
assert load_dotenv('../.env') or load_dotenv('.env')
import sys
sys.path.append(os.environ.get("PROJECTDIR"))
from modules_vae.estimator import VAE
from modules_deepsurv.estimator import DeepSurv
from modules_vae.param_grid import param_grid_exp as param_grid_vae
from modules_deepsurv.param_grid import param_grid_exp as param_grid_deepsurv
from utils.params import VAEParams, DeepsurvParams
from utils.validation import score_external_datasets
from utils.annotate_exp_genes import annotate_exp_genes

import joblib
from dask.distributed import Client, LocalCluster

# tune the hyperparameters of one model on one dataset (either full dataset, or train+valid dataset)
# saves model results like C-index as a json file specified by `params.results_prefix`.json
# if model is exp-only, perform benchmark on external GEO datasets
# scores on these external dataset will also be in the json file
# select the model to be either VAE (default) or Deepsurv (benchmark)
def main(model_name:str='default', 
         endpoint:str='os', 
         shuffle:int=5, 
         fold:int=10, 
         architecture:str='VAE', 
         fulldata:bool=False, 
         subset:bool=False) -> None:
    
    if architecture=='VAE':
        params = VAEParams(model_name=model_name, endpoint=endpoint, shuffle=shuffle, fold=fold, fulldata=fulldata, subset=subset)
        param_grid = param_grid_vae
    elif architecture=='Deepsurv':
        params = DeepsurvParams(model_name=model_name,endpoint=endpoint, shuffle=shuffle, fold=fold, fulldata=fulldata, subset=subset)
        param_grid = param_grid_deepsurv
    else:
        raise NotImplementedError(architecture)
    
    splitsdir=os.environ.get("SPLITDATADIR")
    if fulldata:
        # the model is trained on 100% of the data
        # shuffle and fold are ignored
        # the only use case is for external validation on GEO datasets
        # the validation C-index metric will be set to 0
        train_features_file=f'{splitsdir}/full_features_{endpoint}_processed.parquet'
        train_labels_file=f'{splitsdir}/full_labels.parquet'
        assert os.path.exists(train_features_file) and os.path.exists(train_labels_file)
        train_features=pd.read_parquet(train_features_file)
        train_labels=pd.read_parquet(train_labels_file)[[params.eventcol,params.durationcol]]
        params = annotate_exp_genes(train_features, params)
        train_dataframe=pd.concat([train_labels,train_features],axis=1)
    else:
        # the default mode
        # the model is trained on 80% of the data
        # the 20% validation data is used as calculate hold out C-index metrics
        train_features_file=f'{splitsdir}/{params.shuffle}/{params.fold}/train_features_{endpoint}_processed.parquet'
        train_labels_file=f'{splitsdir}/{params.shuffle}/{params.fold}/train_labels.parquet'
        assert os.path.exists(train_features_file) and os.path.exists(train_labels_file)
        valid_features_file=f'{splitsdir}/{params.shuffle}/{params.fold}/valid_features_{endpoint}_processed.parquet'
        valid_labels_file=f'{splitsdir}/{params.shuffle}/{params.fold}/valid_labels.parquet'
        assert os.path.exists(valid_features_file) and os.path.exists(valid_labels_file)
        train_features=pd.read_parquet(train_features_file)
        train_labels=pd.read_parquet(train_labels_file)[[params.eventcol,params.durationcol]]
        valid_features=pd.read_parquet(valid_features_file)
        valid_labels=pd.read_parquet(valid_labels_file)[[params.eventcol,params.durationcol]]
        params = annotate_exp_genes(train_features, params)
        train_dataframe=pd.concat([train_labels,train_features],axis=1)
        valid_dataframe=pd.concat([valid_labels,valid_features],axis=1)

    if architecture=='VAE':
        # lazy input dims and subtask input dims is already built into VAE estimator.py
        base_estimator = VAE(eventcol=params.eventcol,durationcol=params.durationcol)
    elif architecture=='Deepsurv':
        # lazy input dims is already built into Deepsurv
        base_estimator = DeepSurv(eventcol=params.eventcol,durationcol=params.durationcol)
    else:
        raise NotImplementedError(architecture)
    
    grid_search = GridSearchCV(base_estimator, param_grid)

    cluster = LocalCluster()
    client = Client(cluster)
    with joblib.parallel_config("dask", n_jobs=20): # set n_jobs to NCPUS
        grid_search.fit(train_dataframe)

    # update params with best params
    for k,v in grid_search.best_params_.items():
        setattr(params,k,v)
    # update params with RNA-Seq gene names
    # this field is needed in score_external_datasets
    if params.subset_microarray:
        # only microarray compatible genes were used
        setattr(params,"genes",grid_search.best_estimator_.genes)
    else:
        # all genes were used, including those not found in microarray
        setattr(params,"genes",params.all_exp_genes)
        
    results = {}
    # drop exp genes because it is too long
    # but do print out the final genes used
    results['params_fixed'] = {k: v for k, v in vars(params).items() if not k.startswith('_') and k != 'all_exp_genes' and k not in param_grid.keys()}
    results['params_search'] = {k: v.__str__() for k, v in param_grid.items() } # save activation as string
    results['best_epoch'] = {}
    results['best_epoch']['params'] = {k:v.__str__() for k, v in grid_search.best_params_.items()} # save activation as string

    # skip if this model is for external validation
    if params.fulldata:
        results['best_epoch']['valid_metric'] = 0
    else:
        valid_metric = grid_search.score(valid_dataframe)
        results['best_epoch']['valid_metric'] = valid_metric

    # score external datasets if using only RNASeq as input
    if grid_search.best_estimator_.input_types_all==['exp','clin'] or \
         grid_search.best_estimator_.input_types_all==['exp'] :
        # pass in the torch.nn.Module
        try:
            best_estimator = grid_search.best_estimator_.model.net # DeepSurv estimator
        except AttributeError:
            best_estimator = grid_search.best_estimator_.model # VAE estimator
        
        cindex_uams, cindex_hovon, cindex_emtab, cindex_apex = score_external_datasets(best_estimator,params)
        results['best_epoch']['uams_metric'] = cindex_uams
        results['best_epoch']['hovon_metric'] = cindex_hovon
        results['best_epoch']['emtab_metric'] = cindex_emtab
        results['best_epoch']['apex_metric'] = cindex_apex
        results['best_epoch']['timestamp'] = datetime.now().__str__()
        
    os.makedirs(os.path.dirname(params.resultsprefix),exist_ok=True)

    # save results
    with open(f'{params.resultsprefix}.json', 'w') as f:
        json_dump(results, f, indent=4)

    # save model state dict
    # either estimator class or model class should implement `save`
    grid_search.best_estimator_.save(f'{params.resultsprefix}.pth')

if __name__ == "__main__":
    parser = ArgumentParser(description='Tune hyperparameters of VAE model using scikit-learn GridSearchCV. For adjusting hyperparameters, modify params.py and param_grid.py')
    parser.add_argument('-m','--model_name', type=str, default='exp', help='An experiment name for the model')
    parser.add_argument('-e','--endpoint', type=str, choices=['pfs', 'os','both'], default='both', help='Survival endpoint (pfs or os or both)')
    parser.add_argument('-a', '--architecture', type=str, choices=['VAE','Deepsurv'], default='VAE', help='Choice of model architecture. In-house VAE or comparator Deepsurv.')
    parser.add_argument('-f', '--fulldata', action='store_true', help='Whether to train with full CoMMpass dataset')
    parser.add_argument('-s', '--subset', action='store_true', help='Whether to subset to ensembl genes that have matching microarray probes')
    args = parser.parse_args()
    _pbs_array_id = int(os.getenv('PBS_ARRAY_INDEX', "-1"))
    pbs_shuffle=_pbs_array_id%10
    pbs_fold=_pbs_array_id//10
    if args.endpoint=="both":
        # useful for training the train-valid splits
        # because scheduler has a limit of 99 jobs
        main(args.model_name, 'os', pbs_shuffle, pbs_fold, args.architecture, args.fulldata, args.subset)
        main(args.model_name, 'pfs', pbs_shuffle, pbs_fold, args.architecture, args.fulldata, args.subset)
    else:
        main(args.model_name, args.endpoint, pbs_shuffle, pbs_fold, args.architecture, args.fulldata, args.subset)