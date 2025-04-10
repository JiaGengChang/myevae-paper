from torch.nn import LeakyReLU, ReLU, Sigmoid, Tanh

param_grid = {
    'z_dim': [8, 16, 32],
    'lr': [5e-4, 1e-4, 5e-5], 
    'batch_size': [128, 256, 512],
    'input_types': [['exp','cna']],
    'input_types_subtask': [['clin']],
    'layer_dims': [[[64, 16], [16, 8]], [[128, 32], [64, 16]], [[256, 64], [128, 32]]],
    'layer_dims_subtask' : [[4,1], [8,1], [16,1]],
    'kl_weight': [1],
    'activation': [LeakyReLU(),ReLU(),Sigmoid()],
    'subtask_activation': [Tanh(), Sigmoid()],
    'epochs': [300],
    'burn_in': [50],
    'patience': [20],
    'subset_microarray': [False], # want to do this only for full models
    'scale_method': ['std'] # only for external validation with full model
}

# for debugging uses
param_grid_debug = {
    'z_dim': [8],
    'lr': [5e-4], 
    'batch_size': [1024],
    'input_types': [['exp']],
    'input_types_subtask': [['clin']],
    'layer_dims': [[[32, 4]]],
    'layer_dims_subtask' : [[4,1]],
    'kl_weight': [1],
    'activation': [LeakyReLU()],
    'subtask_activation': [Tanh()],
    'epochs': [100],
    'burn_in': [20],
    'patience': [5],
    'subset_microarray': [True, False], # want to do this only for full models
    'scale_method': ['std'] # only for external validation with full model
}