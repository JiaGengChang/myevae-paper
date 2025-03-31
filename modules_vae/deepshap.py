import torch
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from torch.utils.data import DataLoader
import shap
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
parser = argparse.ArgumentParser(description='Performs DeepSHAP on a VAE model trained using params.py')
parser.add_argument('--endpoint', type=str, choices=['pfs', 'os'], default='os', help='Survival endpoint (pfs or os)')
parser.add_argument('--shuffle', type=int, choices=range(10), default=0, help='Random state for k-fold splitting (0-9)')
parser.add_argument('--fold', type=int, choices=range(5), default=0, help='Fold index for k-fold splitting (0-4)')
args = parser.parse_args([])

# comment out these 3 lines if not using PBS
# PBS array ID to override args.shuffle and args.fold
# _pbs_array_id = int(os.getenv('PBS_ARRAY_INDEX', "-1"))
# args.shuffle=_pbs_array_id%10
# args.fold=_pbs_array_id//10

params = specify_params_here(args.endpoint, args.shuffle, args.fold)

full_dataframe = parse_all(params.endpoint)
train_dataframe, valid_dataframe = kfold_split(full_dataframe, params.shuffle, params.fold)
train_dataframe_scaled, valid_dataframe_scaled = scale_impute(train_dataframe, valid_dataframe, method=params.scaler)

traindata = Dataset(train_dataframe_scaled, params.input_types_all)
validdata = Dataset(valid_dataframe_scaled, params.input_types_all)

model = Model(params.input_types,
              params.input_dims,
              params.layer_dims,
              params.input_types_subtask,
              params.input_dims_subtask,
              params.layer_dims_subtask,
              params.z_dim,
              shap=True)

# load model state dict
model.load_state_dict(torch.load(f'{params.resultsprefix}.pth'))

# background dataset to integrate over
# concatenated along the feature dim (dim 1)
background_data = [traindata.X_exp, traindata.X_clin]

# edit code in the SHAP package
# /home/users/nus/e1083772/.localpython/lib/python3.9/site-packages/shap/explainers/_deep/deep_pytorch.py
# set X = [X] / data = [data]
# comment out the line `X = [x.detach().to(self.device) for x in X]`

shap_explainer = shap.DeepExplainer((model, model.risk_predictor[2]), background_data)

shap_data = [validdata.X_exp, validdata.X_clin]

shap_values = shap_explainer.shap_values(shap_data)

idx=0 # index for input type
shap.summary_plot(shap_values[idx][:,:,0],
                  shapinputs[idx],
                  trainloader.dataset.featurenames_list[idx],
                  max_display=5,
                  plot_size=(6,4)
                 )