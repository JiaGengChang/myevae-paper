import os
import argparse
import torch
import shap 
import matplotlib.pyplot as plt
import numpy as np

import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT)

from modules_vae.model import ShapMultiModalVAE as Model
from utils.dataset import Dataset
from utils.params import VAEParams as SilentParams
from utils.parsers import parse_all, parse_sbs, parse_sv, parse_fish, parse_gistic
from utils.splitter import kfold_split
from utils.scaler import scale_and_impute_without_train_test_leak as scale_impute

"""
 Parse the 3 arguments which we will parallelize across. 
 the actual hyperparameters to modify are in params.py
"""
params = SilentParams()
os.makedirs(os.path.dirname(params.resultsprefix), exist_ok=True) # prepare output directory

full_dataframe = parse_all(params.endpoint)
train_dataframe, valid_dataframe = kfold_split(full_dataframe, params.shuffle, params.fold)
train_dataframe_scaled, valid_dataframe_scaled = scale_impute(train_dataframe, valid_dataframe, method=params.scale_method)

input_types_all = params.input_types + params.input_types_subtask
traindataset = Dataset(train_dataframe_scaled, input_types_all)
validdataset = Dataset(valid_dataframe_scaled, input_types_all)

input_dims = [getattr(traindataset, f'X_{input_type}').shape[1] for input_type in params.input_types]
input_dims_subtask = [getattr(traindataset, f'X_{input_type}').shape[1] for input_type in params.input_types_subtask]
model = Model(
    input_types=params.input_types,
    input_dims=input_dims,
    layer_dims=params.layer_dims,
    input_types_subtask=params.input_types_subtask,
    input_dims_subtask=input_dims_subtask,
    layer_dims_subtask=params.layer_dims_subtask,
    z_dim=params.z_dim,
    activation=torch.nn.LeakyReLU(),
    subtask_activation=torch.nn.Tanh(),
)

# load model state dict
model_checkpoint = torch.load(f'{params.resultsprefix}.pth')
model.load_state_dict(model_checkpoint)

# data ~ list of tensors
background_data = [ getattr(validdataset, f"X_{t}") for t in params.input_types + params.input_types_subtask]
shap_data = [ getattr(traindataset, f"X_{t}") for t in params.input_types + params.input_types_subtask]

explainer = shap.DeepExplainer(model, background_data)

shap_values = explainer.shap_values(shap_data, check_additivity=False)

# assign feature and shap dfs to global env
X_by_type = {}
S_by_type = {}
for (i,t) in enumerate(params.input_types + params.input_types_subtask):
    X_by_type[t] = shap_data[i]
    S_by_type[t] = shap_values[i][:,:,0]

if 'clin' in params.input_types_subtask:
    plt.clf()
    shap.summary_plot(S_by_type['clin'], X_by_type['clin'], plot_type="dot", alpha=0.5, feature_names=['Age','ISS 1','ISS 2','ISS 3','sexIsMale'])
    plt.title('Clinical features')
    plt.tight_layout()
    plt.savefig(f'{params.resultsprefix}_shap_clin.png')

if 'ig' in params.input_types:
    plt.clf()
    sv_names = parse_sv().iloc[:,1:].columns.str.replace('Feature_SeqWGS_','').str.replace('_CALL','')
    shap.summary_plot(S_by_type['ig'], X_by_type['ig'], plot_type="dot", alpha=0.5, feature_names=sv_names)
    plt.title('IgH translocation partners')
    plt.tight_layout()
    plt.savefig(f'{params.resultsprefix}_shap_sv.png')

if 'exp' in params.input_types:
    plt.clf()
    rnaseq_names = np.loadtxt(os.environ.get('RNASEQ_GENE_SYMBOL'),dtype=str)
    shap.summary_plot(S_by_type['exp'], X_by_type['exp'], alpha=0.5, plot_type="dot", max_display=50, feature_names=rnaseq_names)
    plt.title('RNA-Seq gene expression tpm (top 50)')
    plt.tight_layout()
    plt.savefig(f'{params.resultsprefix}_shap_rnaseq_all.png')

if 'cna' in params.input_types:
    plt.clf()
    cna_names = np.loadtxt(os.environ.get('CNA_GENE_SYMBOL'),dtype=str)
    shap.summary_plot(S_by_type['cna'], X_by_type['cna'], alpha=0.5, plot_type="dot", max_display=10, feature_names=cna_names)
    plt.title('Gene-level copy number status (top 10)')
    plt.tight_layout()
    plt.savefig(f'{params.resultsprefix}_shap_cna_all.png')

if 'sbs' in params.input_types:
    plt.clf()
    sbs_names = parse_sbs().iloc[:,1:].columns.str.replace('Feature_','')
    shap.summary_plot(S_by_type['sbs'], X_by_type['sbs'], alpha=0.5, plot_type="dot", max_display=10, feature_names=sbs_names)
    plt.title('SBS Mutation signatures')
    plt.tight_layout()
    plt.savefig(f'{params.resultsprefix}_shap_sbs.png')

if 'fish' in params.input_types:
    plt.clf()
    fish_names = parse_fish().iloc[:,1:].columns.str.replace('Feature_fish_SeqWGS_Cp_','')
    shap.summary_plot(S_by_type['fish'], X_by_type['fish'], alpha=0.5, plot_type="dot", max_display=10, feature_names=fish_names)
    plt.title('WGS iFISH probes copy number status (top 10)')
    plt.tight_layout()
    plt.savefig(f'{params.resultsprefix}_shap_fish.png')

if 'gistic' in params.input_types:
    plt.clf()
    gistic_names = parse_gistic().iloc[:,1:].columns.str.replace('Feature_CNA_','')
    shap.summary_plot(S_by_type['gistic'], X_by_type['gistic'], alpha=0.5, plot_type="dot", max_display=10, feature_names=gistic_names)
    plt.title('GISTIC recurrently amplified/deleleted regions (top 10)')
    plt.tight_layout()
    plt.savefig(f'{params.resultsprefix}_shap_gistic.png')
