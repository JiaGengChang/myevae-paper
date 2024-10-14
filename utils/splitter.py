
from sklearn.model_selection import KFold

# get train and validation data for a given endpoint, shuffle number, and fold number
def kfold_split(full_dataset, random_state, k):
    # randomstate is 0-9 because we are doing 10 repeats
    # k is the fold number, 0-4
    assert k < 5 and k >= 0, "k must be between 0 and 4"
    # all patients, some with missing information e.g. missing RNA-Seq data
    # patients with complete information are suitable as validation examples
    validation_ids = full_dataset.dropna().index

    # pick the k-th fold
    # corresponds to 121 or 122 patients for validation each time
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    for i, (train_idxs, test_idxs) in enumerate(kf.split(validation_ids)):
        if i == k:
            train_ids = validation_ids[train_idxs]
            valid_ids = validation_ids[test_idxs]
            break
    
    # train_dataset is everything minus the validation patients
    # 992 or 993 patients each time
    train_dataset = full_dataset[~full_dataset.index.isin(valid_ids)]
    valid_dataset = full_dataset.loc[valid_ids]

    # warning: train_dataset contains missing values in the form of NaN
    # subsequent code will have to handle this before passing into neural network
    return train_dataset, valid_dataset