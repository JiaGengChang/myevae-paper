from argparse import ArgumentParser
import pandas as pd
import numpy as np
import warnings
warnings.formatwarning = lambda msg, *args, **kwargs: f'{msg}\n'
from sksurv.util import Surv
from json import dump as json_dump
from datetime import datetime

import os
from dotenv import load_dotenv
assert load_dotenv("../.env") or load_dotenv(".env")
import sys
sys.path.append(os.environ.get("PROJECTDIR"))
from utils.params import CoxPHParams
from utils.parsers_external import *
from utils.coxph import create_baseline_model

def main(endpoint, shuffle, fold, fulldata, use_clin):
    params = CoxPHParams(model_name='to_replace',endpoint=endpoint,shuffle=shuffle,fold=fold,fulldata=fulldata)
    splitsdir=os.environ.get("SPLITDATADIR")
    if fulldata:
        # the model is trained on 100% of the data
        # shuffle and fold are ignored
        # the only use case is for external validation on GEO datasets
        # the validation C-index metric will be set to 0
        train_features_file=f'{splitsdir}/full_features_{endpoint}_processed.parquet'
        train_labels_file=f'{splitsdir}/full_labels.parquet'
    else:
        # the default mode
        # the model is trained on 80% of the data
        # the 20% validation data is used as calculate hold out C-index metrics
        train_features_file=f'{splitsdir}/{params.shuffle}/{params.fold}/train_features_{endpoint}_processed.parquet'
        train_labels_file=f'{splitsdir}/{params.shuffle}/{params.fold}/train_labels.parquet'
        valid_features_file=f'{splitsdir}/{params.shuffle}/{params.fold}/valid_features_{endpoint}_processed.parquet'
        valid_labels_file=f'{splitsdir}/{params.shuffle}/{params.fold}/valid_labels.parquet'
        assert os.path.exists(valid_features_file) and os.path.exists(valid_labels_file)
        valid_features=pd.read_parquet(valid_features_file)
        valid_labels=pd.read_parquet(valid_labels_file)[[params.eventcol,params.durationcol]]
        valid_dataframe=pd.concat([valid_labels,valid_features],axis=1)

    assert os.path.exists(train_features_file) and os.path.exists(train_labels_file)
    train_features=pd.read_parquet(train_features_file)
    train_labels=pd.read_parquet(train_labels_file)[[params.eventcol,params.durationcol]]
    
    # Parse survival information
    # Minimum-offset to ensure survival duration is positive, which is required by Cox PH
    train_labels[f'{endpoint}cdy'] -= min(0, train_labels[f'{endpoint}cdy'].min())
    train_y = Surv.from_dataframe(f"cens{endpoint}",f"{endpoint}cdy",train_labels)
    uams_y = Surv.from_arrays(*parse_surv_uams(endpoint))
    hovon_y = Surv.from_arrays(*parse_surv_hovon(endpoint))
    emtab_y = Surv.from_arrays(*parse_surv_emtab(endpoint))
    apex_y = Surv.from_arrays(*parse_surv_apex(endpoint))

    if not fulldata:
        print('Parsing CoMMpass validation set')
        valid_labels[f'{endpoint}cdy'] -= min(0, valid_labels[f'{endpoint}cdy'].min())
        valid_y = Surv.from_dataframe(f"cens{endpoint}",f"{endpoint}cdy",valid_labels)
    
    # Load the GEP risk scores
    commpass_riskscores = pd.read_csv(os.environ.get("COMMPASSRISKSCOREFILE"),index_col=0)
    uams_riskscores = pd.read_csv(os.environ.get("UAMSRISKSCOREFILE"),index_col=0)
    hovon_riskscores = pd.read_csv(os.environ.get("HOVONRISKSCOREFILE"),index_col=0)
    emtab_riskscores = pd.read_csv(os.environ.get("EMTABRISKSCOREFILE"),index_col=0)
    apex_riskscores = pd.read_csv(os.environ.get("APEXRISKSCOREFILE"),index_col=0)
    
    # Create the X data
    train_clin = train_features.filter(regex='Feature_clin')
    uams_clin = parse_clin_uams()
    hovon_clin = parse_clin_hovon()
    emtab_clin = parse_clin_emtab()
    apex_clin = parse_clin_apex()

    X_uams = uams_clin.join(uams_riskscores)
    X_hovon = hovon_clin.join(hovon_riskscores)
    X_emtab = emtab_clin.join(emtab_riskscores)
    X_apex = apex_clin.join(apex_riskscores)
    X_train = train_clin.join(commpass_riskscores)

    # align the column names of df's used for fit and transform
    X_train.columns = X_uams.columns

    if not fulldata:
        valid_clin = valid_features.filter(regex='Feature_clin')
        valid_df = valid_clin.join(commpass_riskscores)
        valid_df.columns = X_uams.columns
        
    # score all models on all datasets
    # for individual models results
    template = {}
    # metadata. main args etc
    template['endpoint'] = endpoint
    template['shuffle'] = shuffle
    template['fold'] = fold
    template['fulldata'] = fulldata
    template['subset'] = params.subset
    template['timestamp'] = datetime.now().__str__()
    template['use_clin'] = use_clin

    # final model names in results
    # their results.json files will be written to separate output folders
    modelnames = ['Clin_only','GEP_UAMS70','GEP_SKY92','GEP_IFM15','GEP_MRC-IX-6']

    for _mname in modelnames:
        if not _mname.startswith('GEP') and not use_clin:
            continue

        results = template.copy()
        results["best_epoch"]={} # for compatibility with other models

        _fname = 'Feature_' + _mname # pattern to match GEP feature column
        # ClinOnly should be a pattern that no GEP features will match
        _model = create_baseline_model(_fname,use_clin)
        # train on CoMMpass training data
        _model.fit(X_train, train_y)
        
        # internal validation set, if using
        if not fulldata:
            _commpass = _model.score(valid_df, valid_y).item()
            results["best_epoch"]["valid_metric"] = _commpass
        else:
            results["best_epoch"]["valid_metric"] = 0.0
            
        # score on various datasets
        _uams = _model.score(X_uams, uams_y).item()
        _hovon = _model.score(X_hovon, hovon_y).item()
        _emtab = _model.score(X_emtab, emtab_y).item()
        _apex = _model.score(X_apex, apex_y).item()
        
        results["best_epoch"]["uams_metric"] =  _uams
        results["best_epoch"]["hovon_metric"] = _hovon
        results["best_epoch"]["emtab_metric"] =  _emtab
        results["best_epoch"]["apex_metric"] = _apex

        resultsprefix = params.resultsprefix.replace('to_replace',_mname+'_noclin' if not use_clin else '')
        os.makedirs(os.path.dirname(resultsprefix),exist_ok=True)

        with open(resultsprefix + '.json','w') as f:
            json_dump(results, f, indent=4)
        

if __name__ == "__main__":
    parser = ArgumentParser(description='Score CoxPH models using GEP risk scores on all datasets')
    parser.add_argument('-e', '--endpoint', type=str, choices=['pfs','os','both'], default='both', help='Survival endpoint (pfs or os or both)')
    parser.add_argument('-f', '--fulldata', action='store_true', help='Whether to train with full CoMMpass dataset')
    parser.add_argument('-c', '--use_clin', action='store_true', help='Whether to train with clinical features - age, sex, ISS.')
    args = parser.parse_args()
    _pbs_array_id = int(os.getenv('PBS_ARRAY_INDEX', "-1"))
    pbs_shuffle=_pbs_array_id%10
    pbs_fold=_pbs_array_id//10
    if args.endpoint=="both":
        main('os', pbs_shuffle, pbs_fold, args.fulldata, args.use_clin)
        main('pfs', pbs_shuffle, pbs_fold, args.fulldata, args.use_clin)
    else:
        main(args.endpoint, pbs_shuffle, pbs_fold, args.fulldata, args.use_clin)