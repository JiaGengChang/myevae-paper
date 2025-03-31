import os
import argparse
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

import sys
sys.path.append('/home/users/nus/e1083772/cancer-survival-ml/')
from utils.preprocessing import *

def main():
    parser = argparse.ArgumentParser(description='Select significant features and preprocess them')
    parser.add_argument('--endpoint', type=str, choices=['pfs', 'os'], default='pfs', help='Survival endpoint (pfs or os)')
    args = parser.parse_args()

    _pbs_array_id = int(os.getenv('PBS_ARRAY_INDEX', "-1"))
    shuffle=_pbs_array_id%10
    fold=_pbs_array_id//10
    endpoint = args.endpoint

    scratchdir="/scratch/users/nus/e1083772/cancer-survival-ml/data/splits"
    features_file=f'{scratchdir}/{shuffle}/{fold}/train_features.parquet'
    features = pd.read_parquet(features_file)

    valid_features_file=f'{scratchdir}/{shuffle}/{fold}/valid_features.parquet'
    valid_features = pd.read_parquet(valid_features_file)

    survcols = [f'{endpoint}cdy',f'cens{endpoint}']
    train_surv_file=f'{scratchdir}/{shuffle}/{fold}/train_labels.parquet'
    train_surv = pd.read_parquet(train_surv_file,columns=survcols)
    train_surv.rename(columns={f'{endpoint}cdy':'survtime',f'cens{endpoint}':'survflag'},inplace=True)

    train_out_features_file=f'{scratchdir}/{shuffle}/{fold}/train_features_processed.parquet'
    valid_out_features_file=f'{scratchdir}/{shuffle}/{fold}/valid_features_processed.parquet'
    
    transformer_gene_exp = Pipeline([
        ('Non-zero variance', VarianceSelector(threshold=0)),
        ('Log1p', Log1pTransform()),
        ('Standard scaling', StandardTransform()),
        ('Cox ElasticNet', CoxnetSelector(l1_ratio=0.5, coef_threshold=0.05)),
    ])

    transformer_gene_cn = Pipeline([
        ('Non-zero variance', VarianceSelector(threshold=0)),
        ('Coxnet', CoxnetSelector(l1_ratio=0.5, coef_threshold=0.05)),
        ('Uncorrelated', CorrelationSelector(threshold=0.9)),
    ])

    transformer_gistic = Pipeline([
        ('Non-zero variance', VarianceSelector(threshold=0)),
        ('Coxnet', CoxnetSelector(l1_ratio=0.5, coef_threshold = 0.1)),
    ])

    transformer_sbs = Pipeline([
        ('Non-zero variance', VarianceSelector(threshold=1)),
        ('Log1p', Log1pTransform()),
        ('Standard scaling', StandardTransform()),
        ('Cox LASSO', CoxnetSelector(l1_ratio=1, coef_threshold=0.1)),
    ])

    transformer_fish = Pipeline([
        ('Non-zero variance', VarianceSelector(threshold=0)),
        ('Coxnet', CoxnetSelector(l1_ratio=0.5, coef_threshold = 0.1)),
    ])

    transformer_ig = Pipeline([
        ('Frequency', FrequencySelector(minfreq=0.01))
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
    ], remainder='passthrough').set_output(transform="pandas")

    algo_args = {
        'n_estimators': 10,
        'max_samples': 0.75,
        'max_features': 0.75,
        'n_jobs':4,
        'verbose':1,
    }
    imputer_args = {
        'max_iter': 10,
        'tol':1e-2,
        'skip_complete':True
    }

    # for continuous variables
    RegressionImputer = IterativeImputer(estimator=RandomForestRegressor(**algo_args), 
                                         initial_strategy='mean', 
                                         **imputer_args)
    # for categorical variables
    ClassificationImputer = IterativeImputer(estimator=RandomForestClassifier(**algo_args),
                                             initial_strategy='most_frequent', 
                                             **imputer_args)

    imputer = ColumnTransformer([
        ('Continuous variables', RegressionImputer, make_column_selector(pattern='Feature_(exp|clin_D_PT_age|SBS)')),
        ('Categorical variables', ClassificationImputer, make_column_selector(pattern='Feature_(?!exp|clin_D_PT_age|SBS)'))
    ], remainder='drop').set_output(transform="pandas")

    # not used for now
    pca = ColumnTransformer([
        ('Gene expression', PCATransform(prefix='Feature_exp',n_components=10), make_column_selector(pattern='Feature_exp')),
        ('Gene copy number', PCATransform(prefix='Feature_CN_gene',n_components=10), make_column_selector(pattern='Feature_CNA_ENSG')),
        ('Gistic copy number', PCATransform(prefix='Feature_CN_gistic',n_components=10), make_column_selector(pattern='Feature_CNA_(RNASeq|SeqWGS)')),
        ('FISH copy number', PCATransform(prefix='Feature_CN_fish',n_components=10), make_column_selector(pattern='Feature_fish')),
    ], remainder='passthrough').set_output(transform="pandas")

    pipeline = Pipeline([
        ('Feature selection', transformer),
        ('Joint imputation', imputer),
        # ('PCA', pca) # optional, can be done later
    ])
    
    # need to shift start date because some OS is negative
    event = train_surv.survflag
    time = train_surv.survtime
    offset = max(0, -np.min(time))
    time += offset
    train_y = Surv.from_arrays(event,time)
    
    out = pipeline.fit_transform(features, train_y)
    outv = pipeline.transform(valid_features)
    
    out.to_parquet(train_out_features_file)
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
    main()