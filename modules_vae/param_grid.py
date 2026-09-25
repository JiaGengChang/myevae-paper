from torch.nn import LeakyReLU, ReLU, Sigmoid, Tanh

param_grid = {
    'z_dim': [8, 16, 32],
    'lr': [5e-4, 1e-4, 5e-5], 
    'batch_size': [128, 256, 512],
    'input_types': [['exp','cna', 'gistic', 'fish', 'sbs', 'ig']],
    'masking_proportions': [
        [{'exp': 0.1,'cna': 0.1,'gistic': 0.1,'fish': 0.1,'sbs': 0.1,'ig': 0.125,}], # 10% or Ig 1/8
        [{'exp': 0.2,'cna': 0.2,'gistic': 0.2,'fish': 0.2,'sbs': 0.2,'ig': 0.25,}], # 20% or Ig 2/8
        [{'exp': 0.3,'cna': 0.3,'gistic': 0.3,'fish': 0.3,'sbs': 0.3,'ig': 0.375,}] # 30% or Ig 3/8
    ],
    'input_types_subtask': [['clin']],
    'layer_dims': [[[64, 16], [16, 8], [16, 4], [16, 4], [4]], 
                   [[128, 32], [64, 16], [16, 8], [16, 4], [4]], 
                   [[256, 64], [128, 32], [32, 8], [16, 4], [4]]],
    'layer_dims_subtask' : [[4,1], [8,1], [16,1]],
    'kl_weight': [1],
    'activation': [LeakyReLU()],
    'subtask_activation': [Tanh()],
    'epochs': [300],
    'burn_in': [50],
    'patience': [20],
}

param_grid_exp_cna_gistic_fish_sbs_ig_chrom = {
    'z_dim': [8, 16, 32, 64],
    'lr': [5e-4, 1e-4, 5e-5], 
    'batch_size': [64, 128],
    'input_types': [['exp','cna', 'gistic', 'fish', 'sbs', 'ig']],
    'input_types_subtask': [['cth','clin']],
    'layer_dims': [[[64, 16], [16, 8], [16, 4], [16, 4], [4], [2]], 
                   [[128, 32], [64, 16], [16, 8], [16, 4], [4], [2]], 
                   [[256, 64], [128, 32], [32, 8], [16, 4], [4], [2]]],
    'layer_dims_subtask' : [[4,1], [8,1], [16,1], [32, 8, 1]],
    'kl_weight': [1, 0.33],
    'activation': [LeakyReLU()],
    'subtask_activation': [Tanh()],
    'epochs': [300],
    'burn_in': [50],
    'patience': [20],
}

param_grid_exp_cna_gistic_fish_sbs_ig = {
    'z_dim': [8, 16, 32, 64],
    'lr': [5e-4, 1e-4, 5e-5], 
    'batch_size': [128, 256, 512],
    'input_types': [['exp','cna', 'gistic', 'fish', 'sbs', 'ig']],
    'input_types_subtask': [['clin']],
    'layer_dims': [[[64, 16], [16, 8], [16, 4], [16, 4], [4], [2]], 
                   [[128, 32], [64, 16], [16, 8], [16, 4], [4], [2]], 
                   [[256, 64], [128, 32], [32, 8], [16, 4], [4], [2]]],
    'layer_dims_subtask' : [[4,1], [8,1], [16,1]],
    'kl_weight': [1],
    'activation': [LeakyReLU(),ReLU(),Sigmoid()],
    'subtask_activation': [Tanh()],
    'epochs': [300],
    'burn_in': [50],
    'patience': [20],
}

param_grid_exp_cna_gistic_fish_sbs = {
    'z_dim': [8, 16, 32],
    'lr': [5e-4, 1e-4, 5e-5], 
    'batch_size': [128, 256, 512],
    'input_types': [['exp','cna', 'gistic', 'fish', 'sbs']],
    'input_types_subtask': [['clin']],
    'layer_dims': [[[64, 16], [16, 8], [16, 4], [16, 4], [4]], 
                   [[128, 32], [64, 16], [16, 8], [16, 4], [4]], 
                   [[256, 64], [128, 32], [32, 8], [16, 4], [4]]],
    'layer_dims_subtask' : [[4,1], [8,1], [16,1]],
    'kl_weight': [1],
    'activation': [LeakyReLU(),ReLU(),Sigmoid()],
    'subtask_activation': [Tanh()],
    'epochs': [300],
    'burn_in': [50],
    'patience': [20],
}

param_grid_exp_cna_gistic_fish = {
    'z_dim': [8, 16, 32],
    'lr': [5e-4, 1e-4, 5e-5], 
    'batch_size': [128, 256, 512],
    'input_types': [['exp','cna', 'gistic', 'fish']],
    'input_types_subtask': [['clin']],
    'layer_dims': [[[64, 16], [16, 8], [16, 4], [16, 4]], 
                   [[128, 32], [64, 16], [16, 8], [16, 4]], 
                   [[256, 64], [128, 32], [32, 8], [16, 4]]],
    'layer_dims_subtask' : [[4,1], [8,1], [16,1]],
    'kl_weight': [1],
    'activation': [LeakyReLU(),ReLU(),Sigmoid()],
    'subtask_activation': [Tanh()],
    'epochs': [300],
    'burn_in': [50],
    'patience': [20],
}

param_grid_exp_cna_gistic = {
    'z_dim': [8, 16, 32],
    'lr': [5e-4, 1e-4, 5e-5], 
    'batch_size': [128, 256, 512],
    'input_types': [['exp','cna', 'gistic']],
    'input_types_subtask': [['clin']],
    'layer_dims': [[[64, 16], [16, 8], [16, 4]], [[128, 32], [64, 16], [16, 8]], [[256, 64], [128, 32], [32, 8]]],
    'layer_dims_subtask' : [[4,1], [8,1], [16,1]],
    'kl_weight': [1],
    'activation': [LeakyReLU(),ReLU(),Sigmoid()],
    'subtask_activation': [Tanh()],
    'epochs': [300],
    'burn_in': [50],
    'patience': [20],
}

param_grid_exp_cna = {
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
}

param_grid_exp = {
    'z_dim': [8, 16, 32],
    'lr': [5e-4, 1e-4, 5e-5], 
    'batch_size': [128, 256, 512],
    'input_types': [['exp']],
    'input_types_subtask': [['clin']],
    'layer_dims': [[[64, 16]], [[128, 32]], [[256, 64]]],
    'layer_dims_subtask' : [[4,1], [8,1], [16,1]],
    'kl_weight': [1],
    'activation': [LeakyReLU(),ReLU(),Sigmoid()],
    'subtask_activation': [Tanh(), Sigmoid()],
    'epochs': [300],
    'burn_in': [50],
    'patience': [20],
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
    'subset_microarray': [True], # want to do this only for full models
    'scale_method': ['std'] # only for external validation with full model
}