import numpy as np
import pandas as pd

# scale and impute train examples
def scale_and_impute_without_train_test_leak(train_dataframe, valid_dataframe, method='minmax'):

    # pull out features
    train_features = train_dataframe.filter(regex='^(?:Feature_)')
    valid_features = valid_dataframe.filter(regex='^(?:Feature_)')
    
    # 0-1 Min-Max scaling for everything
    if method=='minmax':
        train_feature_min = train_features.min(axis=0)
        train_feature_max = train_features.max(axis=0)
        train_features_scaled = (train_features - train_feature_min) / (train_feature_max - train_feature_min)
        valid_features_scaled = (valid_features - train_feature_min) / (train_feature_max - train_feature_min)

        # Clipping valid columns to be within 0 and 1
        valid_features_scaled = valid_features_scaled.clip(0, 1)

        # Filling NAs in training data with 0
        train_features_scaled = train_features_scaled.fillna(0)
        
        # valid fillna is only required for shuffle 5 fold 3
        # where RNA-Seq feature ENSG00000212725 has train_min = train_max
        valid_features_scaled = valid_features_scaled.fillna(0)
    elif method=='std':
        train_feature_mu = train_features.mean(axis=0)
        train_feature_sigma = train_features.std(axis=0)
        train_features_scaled = (train_features - train_feature_mu) / train_feature_sigma
        valid_features_scaled = (valid_features - train_feature_mu) / train_feature_sigma
        train_features_scaled = train_features_scaled.fillna(0)
        valid_features_scaled = valid_features_scaled.fillna(0)
    else:
        raise ValueError(f"Unrecognized method: {method}. Supported methods are 'minmax' and 'std'.")
        
    # add back the survival columns
    train_dataframe_scaled = pd.concat([train_dataframe[['survtime','survflag']], train_features_scaled], axis=1)
    valid_dataframe_scaled = pd.concat([valid_dataframe[['survtime','survflag']], valid_features_scaled], axis=1)
    
    return train_dataframe_scaled, valid_dataframe_scaled