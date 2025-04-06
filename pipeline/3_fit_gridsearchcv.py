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
from modules_deepsurv.estimator import Deepsurv
from modules_vae.params import VAEParams
from modules_deepsurv.params import DeepsurvParams
from modules_vae.param_grid import param_grid
from utils.validation import score_external_datasets
from utils.subset_affy_features import subset_to_microarray_genes
from utils.lazy_input_dims import lazy_input_dims
from utils.annotate_exp_genes import annotate_exp_genes

import joblib
from dask.distributed import Client, LocalCluster

# tune the hyperparameters of one model on one dataset (either full dataset, or train+valid dataset)
# saves model results like C-index as a json file specified by `params.results_prefix`.json
# if model is exp-only, perform benchmark on external GEO datasets
# scores on these external dataset will also be in the json file
# select the model to be either VAE (default) or Deepsurv (benchmark)
def main(endpoint:str='os', shuffle:int=5, fold:int=10, architecture:str='VAE', fulldata:bool=False) -> None:
    
    if architecture=='Deepsurv':
        params = DeepsurvParams(endpoint, shuffle, fold, fulldata)
    elif architecture=='VAE':
        params = VAEParams(endpoint, shuffle, fold, fulldata)
    else:
        raise NotImplementedError(architecture)
    
    splitsdir=os.environ.get("SPLITDATADIR")
    if fulldata:
        # the model is trained on 100% of the data
        # shuffle and fold are ignored
        # the only use case is for external validation on GEO datasets
        # the validation C-index metric will be set to 0
        train_features_file=f'{splitsdir}/full_features_{endpoint}_processed.parquet'
        train_labels_file==f'{splitsdir}/full_labels.parquet'
        train_features=pd.read_parquet(train_features_file)
        train_labels=pd.read_parquet(train_labels_file)[[params.eventcol,params.durationcol]]
        # sets params.exp_genes to all RNA genes in train_dataframe
        params = annotate_exp_genes(train_dataframe, params)
        # should genes be subset to microarray genes ?
        if params.subset_microarray:
            train_features,_ = subset_to_microarray_genes(train_features)
        train_dataframe=pd.concat([train_labels,train_features],axis=1)
    else:
        # the default mode
        # the model is trained on 80% of the data
        # the 20% validation data is used as calculate hold out C-index metrics
        train_features_file=f'{splitsdir}/{params.shuffle}/{params.fold}/train_features_{endpoint}_processed.parquet'
        train_labels_file=f'{splitsdir}/{params.shuffle}/{params.fold}/train_labels.parquet'
        valid_features_file=f'{splitsdir}/{params.shuffle}/{params.fold}/valid_features_{endpoint}_processed.parquet'
        valid_labels_file=f'{splitsdir}/{params.shuffle}/{params.fold}/valid_labels.parquet'
        train_features=pd.read_parquet(train_features_file)
        train_labels=pd.read_parquet(train_labels_file)[[params.eventcol,params.durationcol]]
        valid_features=pd.read_parquet(valid_features_file)
        valid_labels=pd.read_parquet(valid_labels_file)[[params.eventcol,params.durationcol]]
        # sets params.exp_genes to all RNA genes in train_dataframe
        params = annotate_exp_genes(train_dataframe, params)
        # should genes be subset to microarray genes ?
        if params.subset_microarray:
            train_features,valid_features = subset_to_microarray_genes(train_features,valid_features)
        train_dataframe=pd.concat([train_labels,train_features],axis=1)
        valid_dataframe=pd.concat([valid_labels,valid_features],axis=1)

    params = lazy_input_dims(train_dataframe, params) # determine input dimensions. params.input_dims
    
    if architecture=='VAE':
        base_estimator = VAE(eventcol=params.eventcol,durationcol=params.durationcol)
    elif architecture=='Deepsurv':
        base_estimator = Deepsurv(eventcol=params.eventcol,durationcol=params.durationcol)
    else:
        raise NotImplementedError(architecture)
    
    grid_search = GridSearchCV(base_estimator, param_grid)

    cluster = LocalCluster()
    client = Client(cluster)
    with joblib.parallel_config("dask", n_jobs=4):
        grid_search.fit(train_dataframe)

    valid_metric = 0 if fulldata else grid_search.score(valid_dataframe)
    results = {}
    results['params_fixed'] = {k: v for k, v in vars(params).items() if not k.startswith('_') and k != 'exp_genes' and k not in param_grid.keys()}
    results['params_search'] = param_grid
    results['best_model'] = {}
    results['best_model']['params'] = grid_search.best_params_
    results['best_model']['valid_metric'] = valid_metric

    # score external datasets if using only RNASeq as input
    if params.input_types_all == ['exp','clin']:
        try:
            params.exp_genes # genes seen by the model. Not necessarily all input genes.
        except AttributeError:
            params.exp_genes=None
        
        cindex_uams, cindex_hovon, cindex_emtab = score_external_datasets(grid_search.best_estimator_.model, params)
        results['best_model']['uams_metric'] = cindex_uams
        results['best_model']['hovon_metric'] = cindex_hovon
        results['best_model']['emtab_metric'] = cindex_emtab

    # save results
    with open(f'{params.resultsprefix}.json', 'w') as f:
        json_dump(results, f, indent=4)

    # save model state dict
    grid_search.best_estimator_.model.save(f'{params.resultsprefix}.pth')

    return 

if __name__ == "__main__":
    parser = ArgumentParser(description='Tune hyperparameters of VAE model using scikit-learn GridSearchCV. For adjusting hyperparameters, modify params.py and param_grid.py')
    parser.add_argument('-e','--endpoint', type=str, choices=['pfs', 'os'], default='pfs', help='Survival endpoint (pfs or os)')
    parser.add_argument('-a', '--architecture', type=str, choices=['VAE','Deepsurv'], default='VAE', help='Choice of model architecture. In-house VAE or comparator Deepsurv.')
    parser.add_argument('-f', '--fulldata', action='store_true', help='Flag to indicate using train + validation data')
    args = parser.parse_args()
    _pbs_array_id = int(os.getenv('PBS_ARRAY_INDEX', "-1"))
    pbs_shuffle=_pbs_array_id%10
    pbs_fold=_pbs_array_id//10
    main(args.endpoint, pbs_shuffle, pbs_fold, args.architecture, args.fulldata)