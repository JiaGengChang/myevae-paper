param_grid = {
    'input_types_all': [['exp','clin']],
    'n_alphas': [10, 50, 100, 500, 1000], 
    'alpha_min_ratio': [0.1, 0.05, 0.01, 0.005, 0.001],
    'l1_ratio': [1e-3, 1e-2, 0.1, 0.5],
    'normalize': [False, True],
}