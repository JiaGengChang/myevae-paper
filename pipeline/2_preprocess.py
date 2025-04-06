import os
from argparse import ArgumentParser
import pandas as pd # requires pyararow, fastparquet
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sksurv.util import Surv
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_selector, ColumnTransformer
from xgboost import XGBRFClassifier, XGBRFRegressor # version < 1.3.2 to allow non-encoded y

from dotenv import load_dotenv
assert load_dotenv('../.env') or load_dotenv('.env')
import sys
sys.path.append(os.environ.get("PROJECTDIR"))
from utils.preprocessing import *

def main(endpoint:str,shuffle:int,fold:int,fulldata:bool) -> None:

    splitsdir=os.environ.get("SPLITDATADIR")
    survcols = [f'{endpoint}cdy',f'cens{endpoint}']

    # endpoint specific features
    # previously PFS was using processed features from OS, which is not optimal
    if fulldata:
        # shuffle and fold are ignored
        # no valid features because full dataset is used as train
        features_file=f'{splitsdir}/full_features.parquet'
        train_surv_file=f'{splitsdir}/full_labels.parquet'
        train_out_features_file=f'{splitsdir}/full_features_{endpoint}_processed.parquet'
    else:
        features_file=f'{splitsdir}/{shuffle}/{fold}/train_features.parquet'

        valid_features_file=f'{splitsdir}/{shuffle}/{fold}/valid_features.parquet'
        valid_features = pd.read_parquet(valid_features_file)

        train_surv_file=f'{splitsdir}/{shuffle}/{fold}/train_labels.parquet'

        train_out_features_file=f'{splitsdir}/{shuffle}/{fold}/train_features_{endpoint}_processed.parquet'
        valid_out_features_file=f'{splitsdir}/{shuffle}/{fold}/valid_features_{endpoint}_processed.parquet'
    
    features = pd.read_parquet(features_file)
    train_surv = pd.read_parquet(train_surv_file,columns=survcols)
    train_surv.rename(columns={f'{endpoint}cdy':'survtime',f'cens{endpoint}':'survflag'},inplace=True)

    transformer_gene_exp = Pipeline([
        ('Non-zero variance', VarianceSelector(threshold=0)),
        ('Log1p', Log1pTransform()),
        ('Standard scaling', StandardTransform()),
        ('Cox ElasticNet', CoxnetSelector(l1_ratio=0.5, coef_threshold=0.05)),
    ])

    transformer_sbs = Pipeline([
        ('Non-zero variance', VarianceSelector(threshold=1)),
        ('Log1p', Log1pTransform()),
        ('Standard scaling', StandardTransform()),
        ('Cox LASSO', CoxnetSelector(l1_ratio=1, coef_threshold=0.1)),
    ])

    transformer_gene_cn = Pipeline([
        ('Non-zero variance', VarianceSelector(threshold=0)),
        ('Coxnet', CoxnetSelector(l1_ratio=0.5, coef_threshold=0.05)),
        ('Uncorrelated', CorrelationSelector(threshold=0.9)),
    ])

    transformer_gistic = Pipeline([
        ('Non-zero variance', VarianceSelector(threshold=0)),
        ('Coxnet', CoxnetSelector(l1_ratio=0.5, coef_threshold = 0.2)),
    ])

    transformer_fish = Pipeline([
        ('Non-zero variance', VarianceSelector(threshold=0)),
        ('Coxnet', CoxnetSelector(l1_ratio=0.5, coef_threshold = 0.2)),
    ])

    transformer_clin = Pipeline([
        ('Scale age', StandardTransform(cols=['Feature_clin_D_PT_age']))
    ])

    transformer_ig = Pipeline([
        ('Frequency', FrequencySelector(minfreq=0.01))
    ])

    transformer = ColumnTransformer([
        ('Gene expression', transformer_gene_exp, make_column_selector(pattern='Feature_exp_')),
        ('Gene copy number', transformer_gene_cn, make_column_selector(pattern='Feature_CNA_ENSG')),
        ('Gistic copy number', transformer_gistic, make_column_selector(pattern='Feature_CNA_(Amp|Del)')),
        ('FISH copy number', transformer_fish, make_column_selector(pattern='Feature_fish')),
        ('Mutation signatures', transformer_sbs, make_column_selector(pattern='Feature_SBS')),
        ('Clinical', transformer_clin, make_column_selector(pattern='Feature_clin')),
    ], remainder='passthrough').set_output(transform="pandas")

    tree_args = {
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bynode': 0.8,
        'n_jobs': 8,
    }
    imputer_args = {
        'n_nearest_features':300,
        'max_iter':50,
        'tol': 0.01,
        'skip_complete':True,
    }

    ContinuousImputer = IterativeImputer(estimator=XGBRFRegressor(**tree_args), initial_strategy='mean', **imputer_args)
    CategoricalImputer = IterativeImputer(estimator=XGBRFClassifier(**tree_args), initial_strategy='most_frequent', **imputer_args)

    imputer = ColumnTransformer([
        ('Continuous variables', ContinuousImputer, make_column_selector(pattern='Feature_(exp|clin_D_PT_age|SBS)')),
        ('Categorical variables', CategoricalImputer, make_column_selector(pattern='Feature_(?!exp|clin_D_PT_age|SBS)'))
    ], remainder='drop').set_output(transform="pandas")

    # pca = ColumnTransformer([
    #     ('Gene expression', PCATransform(prefix='Feature_exp',n_components=10), make_column_selector(pattern='Feature_exp')),
    #     ('Gene copy number', PCATransform(prefix='Feature_CN_gene',n_components=10), make_column_selector(pattern='Feature_CNA_ENSG')),
    #     ('Gistic copy number', PCATransform(prefix='Feature_CN_gistic',n_components=10), make_column_selector(pattern='Feature_CNA_(RNASeq|SeqWGS)')),
    #     ('FISH copy number', PCATransform(prefix='Feature_CN_fish',n_components=10), make_column_selector(pattern='Feature_fish')),
    # ], remainder='passthrough').set_output(transform="pandas")

    pipeline = Pipeline([
        ('Feature selection', transformer),
        ('Joint imputation', imputer),
    ])
    
    # need to shift start date because some OS is negative
    event = train_surv.survflag
    time = train_surv.survtime
    offset = max(0, -np.min(time))
    time += offset
    train_y = Surv.from_arrays(event,time)
    
    out = pipeline.fit_transform(features, train_y)
    out.to_parquet(train_out_features_file)

    if not fulldata:
        outv = pipeline.transform(valid_features)
        outv.to_parquet(valid_out_features_file)
        
    print(f'# significant features remaining:')
    print(f'RNA exp:\t{out.filter(regex="Feature_exp_ENSG").shape[1]} \t out of \t {features.filter(regex="Feature_exp_ENSG").shape[1]}')
    print(f'CN Gene:\t{out.filter(regex="Feature_CNA_ENSG").shape[1]} \t out of \t {features.filter(regex="Feature_CNA_ENSG").shape[1]}')
    print(f'CN Gistic:\t{out.filter(regex="Feature_CNA_(Amp|Del)").shape[1]} \t out of \t {features.filter(regex="Feature_CNA_(Amp|Del)").shape[1]}')
    print(f'FISH:\t\t{out.filter(regex="Feature_fish").shape[1]} \t out of \t {features.filter(regex="Feature_fish").shape[1]}')
    print(f'SBS:\t\t{out.filter(regex="Feature_SBS").shape[1]} \t out of \t {features.filter(regex="Feature_SBS").shape[1]}')
    print(f'IGH trans:\t{out.filter(regex="Feature_SeqWGS").shape[1]} \t out of \t {features.filter(regex="Feature_SeqWGS").shape[1]}')
    print(f'Clinical:\t{out.filter(regex="Feature_clin").shape[1]} \t out of \t {features.filter(regex="Feature_clin").shape[1]}')
    
    return

if __name__ == "__main__":
    parser = ArgumentParser(description='Select significant features and preprocess them')
    parser.add_argument('-e','--endpoint', type=str, choices=['pfs', 'os'], default='pfs', help='Survival endpoint (pfs or os)')
    parser.add_argument('-f','--fulldata', action='store_true', help='Flag indicating whether to use all data')
    args = parser.parse_args()

    _pbs_array_id = int(os.getenv('PBS_ARRAY_INDEX', "-1"))
    pbs_shuffle=_pbs_array_id%10
    pbs_fold=_pbs_array_id//10
    
    main(args.endpoint, pbs_shuffle, pbs_fold, args.fulldata)