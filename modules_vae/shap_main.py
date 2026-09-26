import os
import argparse
import pandas as pd
import torch

import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT)

from modules_vae.estimator import VAE
from utils.params import VAEParams as specify_params_here
from utils.parsers import parse_all
from utils.splitter import kfold_split
from utils.scaler import scale_and_impute_without_train_test_leak as scale_impute

"""
 Parse the 3 arguments which we will parallelize across. 
 the actual hyperparameters to modify are in params.py
"""
parser = argparse.ArgumentParser(description='Train VAE model on the full dataset, for SHAP. For adjusting hyperparameters, modify params.py')
parser.add_argument('--endpoint', type=str, choices=['pfs', 'os'], default='pfs', help='Survival endpoint (pfs or os)')
args = parser.parse_args()

params = specify_params_here(args.endpoint)

os.makedirs(os.path.dirname(params.resultsprefix), exist_ok=True) # prepare output directory

full_dataframe = parse_all(params.endpoint)
train_dataframe, valid_dataframe = kfold_split(full_dataframe, params.shuffle, params.fold)
train_dataframe_scaled, valid_dataframe_scaled = scale_impute(train_dataframe, valid_dataframe, method=params.scale_method, exclude_clin=True)
full_dataframe_scaled = pd.concat([train_dataframe_scaled,valid_dataframe_scaled])
full_dataframe_scaled = full_dataframe_scaled.rename(columns={
    'survtime': params.durationcol,
    'survflag': params.eventcol,
})

model = VAE(
    input_types=params.input_types,
    subset_microarray=params.subset,
    layer_dims=params.layer_dims,
    input_types_subtask=params.input_types_subtask,
    layer_dims_subtask=params.layer_dims_subtask,
    z_dim=params.z_dim,
    lr=params.lr,
    batch_size=params.batch_size,
    epochs=params.epochs,
    burn_in=params.burn_in,
    patience=params.patience,
    eventcol=params.eventcol,
    durationcol=params.durationcol,
    kl_weight=params.kl_weight,
    activation=torch.nn.LeakyReLU(),
    subtask_activation=torch.nn.Tanh(),
    scale_method=params.scale_method,
    masking_proportions=getattr(params, 'masking_proportions', None),
)

model.fit(full_dataframe_scaled)

# save model state dict
model.save(f'{params.resultsprefix}.pth')