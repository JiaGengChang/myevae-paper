param_grid = {
    'z_dim': [8, 16, 32],
    'lr': [5e-4, 1e-4, 5e-5], 
    'batch_size': [32, 64, 128],
    'layer_dims': [[[32]], [[64]], [[128]]],
    'layer_dims_subtask' : [[4,1], [8,1], [16,1]]
}