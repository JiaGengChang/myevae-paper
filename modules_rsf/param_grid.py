param_grid = {
    "input_types_all": [['exp', 'clin']],
    "n_estimators": [50, 100, 200],
    "max_depth": [20, 50, 100],
    "min_samples_split": [5, 6, 7],
    "min_samples_leaf": [3],
    "oob_score": [False, True],
    "max_samples": [None, 0.8]
}