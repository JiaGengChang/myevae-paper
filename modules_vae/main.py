import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from torch.utils.data import DataLoader
import argparse

from params import VAEParams as specify_params_here
import sys
sys.path.append('../utils')
from dataset import Dataset
from model import MultiModalVAE as Model
from parsers import parse_all
from splitter import kfold_split
from scaler import scale_and_impute_without_train_test_leak as scale_impute
from fit import fit
from predict import predict_to_tsv
from plotlosses import plot_results_to_pdf

"""
 Parse the 3 arguments which we will parallelize across. 
 the actual hyperparameters to modify are in params.py
"""
parser = argparse.ArgumentParser(description='Train VAE model. For adjusting hyperparameters, modify params.py')
parser.add_argument('--endpoint', type=str, choices=['pfs', 'os'], default='pfs', help='Survival endpoint (pfs or os)')
parser.add_argument('--shuffle', type=int, choices=range(10), default=0, help='Random state for k-fold splitting (0-9)')
parser.add_argument('--fold', type=int, choices=range(5), default=0, help='Fold index for k-fold splitting (0-4)')
args = parser.parse_args()

# comment out these 3 lines if not using PBS
# PBS array ID to override args.shuffle and args.fold
_pbs_array_id = int(os.getenv('PBS_ARRAY_INDEX', "-1"))
args.shuffle=_pbs_array_id%10
args.fold=_pbs_array_id//10

params = specify_params_here(args.endpoint, args.shuffle, args.fold)

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
predict_to_tsv(model, validloader, f'{params.resultsprefix}.tsv', save_embeddings=True)

# plot losses and metrics to pdf
# plot_results_to_pdf(f'{params.resultsprefix}.json',f'{params.resultsprefix}.pdf')

# save model state dict
# model.save(f'{params.resultsprefix}.pth')