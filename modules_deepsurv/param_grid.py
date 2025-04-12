from torch.nn import *

param_grid_exp_cna = {
    'input_types_all':[['exp','cna','clin']],
    'lr': [5e-4, 1e-4, 5e-5, 1e-5],
    'layer_dims': [[64, 16, 1], [128, 32, 1], [256, 64, 8, 1]],
    'batch_size': [128, 256, 512],
    'dropout': [0.2, 0.3, 0.5],
    'activation': [LeakyReLU(), ReLU(), Sigmoid(), Tanh()],
    'epochs': [300],
    'burn_in': [50],
    'patience': [20],
    'scale_method': ['std'] # only for external validation with full model
}

param_grid_exp = {
    'input_types_all':[['exp','clin']],
    'lr': [5e-4, 1e-4, 5e-5, 1e-5],
    'layer_dims': [[64, 16, 1], [128, 32, 1], [256, 64, 8, 1]],
    'batch_size': [128, 256, 512],
    'dropout': [0.2, 0.3, 0.5],
    'activation': [LeakyReLU(), ReLU(), Sigmoid(), Tanh()],
    'epochs': [300],
    'burn_in': [50],
    'patience': [20],
    'scale_method': ['std'] # only for external validation with full model
}

# for debugging fit_gridsearchcv.py
param_grid_debug = {
    'input_types_all':[['exp','clin']],
    'layer_dims': [[2, 1], [4, 1]],
    'lr': [5e-4],
    'batch_size': [1024],
    'dropout': [0.5],
    'activation': [LeakyReLU()],
    'epochs': [100],
    'burn_in': [10],
    'patience': [10],
    'scale_method': ['std']
}