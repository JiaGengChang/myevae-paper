import numpy as np
import pandas as pd

# scale and impute train examples
def scale_and_impute_without_train_test_leak(train_dataframe, valid_dataframe, method='minmax', exclude_clin=False, regex='^(?:Feature_)'):

    # applies user-specified scaling method only on non-clinical features
    # clinical features will be standardized (age) or left as-is (sex, ISS category)
    if exclude_clin:
        # extract clinical features
        train_dataframe_clin = train_dataframe.filter(regex='^(?:Feature_clin_)')
        valid_dataframe_clin = valid_dataframe.filter(regex='^(?:Feature_clin_)')
        # standardize the age column
        age = 'Feature_clin_D_PT_age'
        train_dataframe_clin[age] = train_dataframe_clin[[age]].apply(lambda x: (x - x.mean()) / x.std()).values
        valid_dataframe_clin[age] = valid_dataframe_clin[[age]].apply(lambda x: (x - x.mean()) / x.std()).values
        # fill the 0 in clinical features
        train_dataframe_clin = train_dataframe_clin.fillna(0)
        valid_dataframe_clin = valid_dataframe_clin.fillna(0)
        # perform user-specified scaling on non-clinical features
        regex_aclin = '^(?:Feature_)(?!clin_)'
        train_dataframe_aclin, valid_dataframe_aclin = scale_and_impute_without_train_test_leak(train_dataframe,valid_dataframe,method=method,exclude_clin=False,regex=regex_aclin)
        # concatenate the clinical and non-clinical features
        train_dataframe_out = pd.concat([train_dataframe_clin, train_dataframe_aclin], axis=1)  
        valid_dataframe_out = pd.concat([valid_dataframe_clin, valid_dataframe_aclin], axis=1)
        return train_dataframe_out, valid_dataframe_out
    
    # pull out features
    train_features = train_dataframe.filter(regex = regex)
    valid_features = valid_dataframe.filter(regex = regex)
    
    # optional perform log(1+x) transform
    if method.startswith('log1p'):
        train_features = np.log1p(train_features)
        valid_features = np.log1p(valid_features)
    
    # 0-1 Min-Max scaling for everything    
    if 'minmax' in method:
        train_feature_min = train_features.min(axis=0)
        train_feature_max = train_features.max(axis=0)
        train_features_scaled = (train_features - train_feature_min) / (train_feature_max - train_feature_min)
        valid_features_scaled = (valid_features - train_feature_min) / (train_feature_max - train_feature_min)
        valid_features_scaled = valid_features_scaled.clip(0, 1)
    elif 'std' in method:
        train_feature_mu = train_features.mean(axis=0)
        train_feature_sigma = train_features.std(axis=0)
        train_features_scaled = (train_features - train_feature_mu) / train_feature_sigma
        valid_features_scaled = (valid_features - train_feature_mu) / train_feature_sigma
    elif 'robust' in method:
        train_features_median = train_features.median(axis=0)
        train_features_iqr = train_features.quantile(0.95) - train_features.quantile(0.05) + 1e-5
        train_features_scaled = (train_features - train_features_median + 1e-5) / train_features_iqr
        valid_features_scaled = (valid_features - train_features_median + 1e-5) / train_features_iqr
    elif 'rank' in method:
        train_features_scaled = train_features.rank(pct=True)
        valid_features_scaled = valid_features.rank(pct=True)
    elif method=='none':
        train_features_scaled = train_features
        valid_features_scaled = valid_features
    else:
        raise ValueError(f"Unrecognized method: {method}. Supported methods are 'minmax','std','robust','rank'.")
        
    # fill 0 for missing values
    train_features_scaled = train_features_scaled.fillna(0)
    # valid fillna is only required for shuffle 5 fold 3
    # where RNA-Seq feature ENSG00000212725 has train_min = train_max
    valid_features_scaled = valid_features_scaled.fillna(0)
        
    # add back the survival columns
    train_dataframe_scaled = pd.concat([train_dataframe[['survtime','survflag']], train_features_scaled], axis=1)
    valid_dataframe_scaled = pd.concat([valid_dataframe[['survtime','survflag']], valid_features_scaled], axis=1)
    
    return train_dataframe_scaled, valid_dataframe_scaled