from torch.nn.activations import LeakyReLU, ReLU, Sigmoid

param_grid = {
    'lr': [5e-4, 1e-4, 5e-5, 1e-5],
    'layer_dims': [[None, 32, 1], [None, 64, 1], [None, 128, 32, 1], [None, 512, 128, 32, 1]],
    'batch_size': [128, 256, 512],
    'dropout': [0.2, 0.3, 0.5],
    'activation': [LeakyReLU(), ReLU(), Sigmoid()],
}