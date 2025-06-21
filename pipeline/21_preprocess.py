import os
from argparse import ArgumentParser
import pandas as pd # requires pyararow, fastparquet
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sksurv.util import Surv
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor 
import sys
sys.path.append('/home/users/nus/e1083772/cancer-survival-ml/utils')
from pipelinetools import *

def main(endpoint:str,
         datadir:str) -> None:

    # oscdy is the time to overall survival
    # censos is the event flag for overall survival
    # pfscdy is the time to progression-free survival
    # censpfs is the event flag for progression-free survival
    survcols = [f'{endpoint}cdy',f'cens{endpoint}']

    features_file=f'{datadir}/train_features.parquet'
    features = pd.read_parquet(features_file)

    train_surv_file=f'{datadir}/train_labels.parquet'
    train_surv = pd.read_parquet(train_surv_file,columns=survcols)
    train_surv.rename(columns={f'{endpoint}cdy':'survtime',f'cens{endpoint}':'survflag'},inplace=True)

    valid_features_file=f'{datadir}/valid_features.parquet'
    valid_features = pd.read_parquet(valid_features_file)

    train_out_features_file=f'{datadir}/train_features_{endpoint}_processed_new.parquet'
    valid_out_features_file=f'{datadir}/valid_features_{endpoint}_processed_new.parquet'    

    transformer_gene_exp = Pipeline([
        ('Non-zero variance', VarianceSelector(threshold=0)),
        ('Log1p', Log1pTransform()),
        ('Standard scaling', StandardTransform()),
        ('Cox ElasticNet', CoxnetSelector(l1_ratio=0.5, coef_threshold=0.05)),
    ])

    transformer_sbs = Pipeline([
        ('Top N selector', TopNSelector(n=10)),
    ])

    transformer_gene_cn = Pipeline([
        ('Non-zero variance', VarianceSelector(threshold=0)),
        ('Coxnet', CoxnetSelector(l1_ratio=0.5, coef_threshold=0.05)),
        ('Uncorrelated', CorrelationSelector(threshold=0.9)),
    ])

    # same as fish
    transformer_gistic = Pipeline([
        ('Non-zero variance', VarianceSelector(threshold=0)),
        ('Coxnet', CoxnetSelector(l1_ratio=0.5, coef_threshold = 0.2)),
    ])
    # same as gistic
    transformer_fish = Pipeline([
        ('Non-zero variance', VarianceSelector(threshold=0)),
        ('Coxnet', CoxnetSelector(l1_ratio=0.5, coef_threshold = 0.2)),
    ])

    transformer_clin = Pipeline([
        ('Scale age', StandardTransform(cols=['Feature_clin_D_PT_age']))
    ])

    transformer = ColumnTransformer([
        ('Gene expression', transformer_gene_exp, make_column_selector(pattern='Feature_exp_')),
        ('Gene copy number', transformer_gene_cn, make_column_selector(pattern='Feature_CNA_ENSG')),
        ('Gistic copy number', transformer_gistic, make_column_selector(pattern='Feature_CNA_(Amp|Del)')),
        ('FISH copy number', transformer_fish, make_column_selector(pattern='Feature_fish')),
        ('Mutation signatures', transformer_sbs, make_column_selector(pattern='Feature_SBS')),
        ('Clinical', transformer_clin, make_column_selector(pattern='Feature_clin')),
    ], remainder='drop').set_output(transform="pandas")

    tree_args = {
        'n_estimators': 100,
        'max_depth': 20,
        'min_samples_split': 5,
        'n_jobs': -1,
    }
    imputer_args = {
        'n_nearest_features':100,
        'max_iter':50,
        'tol': 0.001,
        'skip_complete':True,
    }

    ContinuousImputer = IterativeImputer(estimator=RandomForestRegressor(**tree_args), initial_strategy='mean', **imputer_args)
    CategoricalImputer = IterativeImputer(estimator=RandomForestClassifier(**tree_args), initial_strategy='most_frequent', **imputer_args)

    imputer = ColumnTransformer([
        ('Continuous variables', ContinuousImputer, make_column_selector(pattern='Feature_(exp|clin_D_PT_age|SBS)')),
        ('Categorical variables', CategoricalImputer, make_column_selector(pattern='Feature_(?!exp|clin_D_PT_age|SBS)'))
    ], remainder='drop').set_output(transform="pandas")
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
    
if __name__ == "__main__":
    parser = ArgumentParser(description='Select significant features and preprocess them')
    parser.add_argument('-e','--endpoint', type=str, choices=['pfs', 'os'], help='Survival endpoint to select features against (pfs or os)')
    parser.add_argument('-d', '--datadir', type=str, help='Input directory for model training and validation data')
    args = parser.parse_args()
    
    main(args.endpoint,args.datadir)