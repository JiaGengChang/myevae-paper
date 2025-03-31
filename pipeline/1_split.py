import os
os.chdir('/home/users/nus/e1083772/cancer-survival-ml/')
import argparse
import pandas as pd
import numpy as np

import sys
sys.path.append('/home/users/nus/e1083772/cancer-survival-ml/')
from utils.dataset import Dataset
from utils.parsers import parse_all
from utils.splitter import kfold_split

def main():
    REDO=True # whether to overwrite existing parquet files
    
    _pbs_array_id = int(os.getenv('PBS_ARRAY_INDEX', "-1"))
    shuffle=_pbs_array_id%10
    fold=_pbs_array_id//10

    # begin writing split data
    scratchdir="/scratch/users/nus/e1083772/cancer-survival-ml/data/splits"
    train_features_file=f'{scratchdir}/{shuffle}/{fold}/train_features.parquet'
    valid_features_file=f'{scratchdir}/{shuffle}/{fold}/valid_features.parquet'

    train_labels_file=f'{scratchdir}/{shuffle}/{fold}/train_labels.parquet'
    valid_labels_file=f'{scratchdir}/{shuffle}/{fold}/valid_labels.parquet'

    if not os.path.isfile(train_features_file) \
        or not os.path.isfile(valid_features_file) \
            or not os.path.isfile(train_labels_file) \
                or not os.path.isfile(valid_labels_file) \
                    or REDO:
        full_dataframe = parse_all()
        train_dataframe, valid_dataframe = kfold_split(full_dataframe, shuffle, fold)

        survcols=['oscdy','censos','pfscdy','censpfs']
        train_surv = train_dataframe.loc[:,survcols]
        valid_surv = valid_dataframe.loc[:,survcols]
        train_dataframe.drop(columns=survcols,inplace=True)
        valid_dataframe.drop(columns=survcols,inplace=True)
        
        if not os.path.exists(os.path.dirname(train_features_file)):
            os.makedirs(os.path.dirname(train_features_file), exist_ok=True)
        train_dataframe.to_parquet(train_features_file,compression=None)
        train_surv.to_parquet(train_labels_file, compression=None)
        
        if not os.path.exists(os.path.dirname(valid_features_file)):
            os.makedirs(os.path.dirname(valid_features_file), exist_ok=True)
        valid_dataframe.to_parquet(valid_features_file,compression=None)
        valid_surv.to_parquet(valid_labels_file, compression=None)
    
    return


if __name__ == "__main__":
    main()