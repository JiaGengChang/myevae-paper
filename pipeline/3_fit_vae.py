import os
import argparse
import pandas as pd
import numpy as np

import sys
sys.path.append('/home/users/nus/e1083772/cancer-survival-ml/')
from modules_vae.params import VAEParams as specify_params_here
from modules_vae.fit import fit
from modules_vae.model import MultiModalVAE as Model
from modules_vae.predict import predict_to_tsv

from utils.dataset import Dataset
from utils.parsers import parse_all
from utils.splitter import kfold_split
from utils.scaler import scale_and_impute_without_train_test_leak as scale_impute
from utils.plotlosses import plot_results_to_pdf


def main():
    """
    Parse the 3 arguments which we will parallelize across. 
    the actual hyperparameters to modify are in params.py
    """
    parser = argparse.ArgumentParser(description='Train VAE model. For adjusting hyperparameters, modify params.py')
    parser.add_argument('--endpoint', type=str, choices=['pfs', 'os'], default='pfs', help='Survival endpoint (pfs or os)')
    # parser.add_argument('--shuffle', type=int, choices=range(10), default=0, help='Random state for k-fold splitting (0-9)')
    # parser.add_argument('--fold', type=int, choices=range(5), default=0, help='Fold index for k-fold splitting (0-4)')
    args = parser.parse_args()

    # comment out these 3 lines if not using PBS
    # PBS array ID to override shuffle and fold
    _pbs_array_id = int(os.getenv('PBS_ARRAY_INDEX', "-1"))
    pbs_shuffle=_pbs_array_id%10
    pbs_fold=_pbs_array_id//10
    params = specify_params_here(args.endpoint, pbs_shuffle, pbs_fold)

    # begin writing split data
    scratchdir="/scratch/users/nus/e1083772/cancer-survival-ml/data/splits"
    train_features_file=f'{scratchdir}/{params.shuffle}/{params.fold}/train_features.parquet'
    valid_features_file=f'{scratchdir}/{params.shuffle}/{params.fold}/valid_features.parquet'

    train_labels_file=f'{scratchdir}/{params.shuffle}/{params.fold}/train_labels.parquet'
    valid_labels_file=f'{scratchdir}/{params.shuffle}/{params.fold}/valid_labels.parquet'

    REDO=True # whether to overwrite existing parquet files
    if not os.path.isfile(train_features_file) \
        or not os.path.isfile(valid_features_file) \
            or not os.path.isfile(train_labels_file) \
                or not os.path.isfile(valid_labels_file) \
                    or REDO:
        full_dataframe = parse_all(params.endpoint)
        train_dataframe, valid_dataframe = kfold_split(full_dataframe, params.shuffle, params.fold)

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
    else:
        train_dataframe = pd.read_parquet(train_features_file)
        valid_dataframe = pd.read_parquet(valid_features_file)
        survcols = [f'{params.endpoint}cdy',f'cens{params.endpoint}']
        train_surv = pd.read_parquet(train_labels_file,columns=survcols)
        valid_surv = pd.read_parquet(valid_labels_file,columns=survcols)
        train_surv.rename(columns={f'{params.endpoint}cdy':'survtime',f'cens{params.endpoint}':'survflag'},inplace=True)
        valid_surv.rename(columns={f'{params.endpoint}cdy':'survtime',f'cens{params.endpoint}':'survflag'},inplace=True)
        
    return
    # end of writing split data

    os.makedirs(os.path.dirname(params.resultsprefix), exist_ok=True) # prepare output directory

    full_dataframe = parse_all(params.endpoint)
    train_dataframe, valid_dataframe = kfold_split(full_dataframe, params.shuffle, params.fold)
    train_dataframe_scaled, valid_dataframe_scaled = scale_impute(train_dataframe, valid_dataframe, method=params.scaler)

    trainloader = DataLoader(Dataset(train_dataframe_scaled, params.input_types_all), batch_size=params.batch_size, shuffle=True)
    validloader = DataLoader(Dataset(valid_dataframe_scaled, params.input_types_all), batch_size=128, shuffle=False)

    model = Model(params.input_types,
                params.input_dims,
                params.layer_dims,
                params.input_types_subtask,
                params.input_dims_subtask,
                params.layer_dims_subtask,
                params.z_dim)

    # fit and save history to json
    fit(model, trainloader, validloader, params)

    # predict on validation data once more and save to tsv
    # predict_to_tsv(model, validloader, f'{params.resultsprefix}.tsv', save_embeddings=True)

    # plot losses and metrics to pdf
    # plot_results_to_pdf(f'{params.resultsprefix}.json',f'{params.resultsprefix}.pdf')

    # save model state dict
    model.save(f'{params.resultsprefix}.pth')


if __name__ == "__main__":
    main()