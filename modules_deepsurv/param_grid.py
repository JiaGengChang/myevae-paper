from torch.nn.activations import LeakyReLU, ReLU, Sigmoid

param_grid = {
    'input_types_all':[['exp','clin']],
    'lr': [5e-4, 1e-4, 5e-5, 1e-5],
    'layer_dims': [[32, 1], [64, 1], [128, 32, 1]],
    'batch_size': [128, 256, 512],
    'dropout': [0.2, 0.3, 0.5],
    'activation': [LeakyReLU(), ReLU(), Sigmoid()],
    'epochs': [300],
    'burn_in': [50],
    'patience': [20],
    'subset_microarray': [True, False],
    'scale_method': ['std']
}