from torch.nn import LeakyReLU, ReLU, Sigmoid

param_grid = {
    'input_types_all':[['exp','cna','clin']],
    'lr': [5e-4, 1e-4, 5e-5, 1e-5],
    'layer_dims': [[64, 16, 1], [128, 32, 1], [256, 64, 8, 1]],
    'batch_size': [128, 256, 512],
    'dropout': [0.2, 0.3, 0.5],
    'activation': [LeakyReLU(), ReLU(), Sigmoid()],
    'epochs': [300],
    'burn_in': [50],
    'patience': [20],
    'subset_microarray': [False], # want to do this only for full models
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
    'subset_microarray': [False],
    'scale_method': ['std']
}