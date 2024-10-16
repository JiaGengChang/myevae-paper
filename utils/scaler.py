import numpy as np
import pandas as pd

# scale and impute train examples
def scale_and_impute_without_train_test_leak(train_dataframe, valid_dataframe):

    # Separate scaling for Feature_exp and Feature_clin_D_PT_age
    train_feature_exp_cols = train_dataframe.filter(regex='^Feature_exp')
    train_feature_clin_age_cols = train_dataframe.filter(regex='^Feature_clin_D_PT_age')
    train_noneedscale_cols = train_dataframe.filter(regex='^(?!Feature_exp|Feature_clin_D_PT_age)')
    
    valid_feature_exp_cols = valid_dataframe.filter(regex='^Feature_exp')
    valid_feature_clin_age_cols = valid_dataframe.filter(regex='^Feature_clin_D_PT_age')
    valid_noneedscale_cols = valid_dataframe.filter(regex='^(?!Feature_exp|Feature_clin_D_PT_age)')

    # 0-1 Min-Max scaling
    train_feature_exp_cols_scaled = train_feature_exp_cols
    valid_feature_exp_cols_scaled = valid_feature_exp_cols
    train_feature_exp_min = train_feature_exp_cols_scaled.min(axis=0)
    train_feature_exp_max = train_feature_exp_cols_scaled.max(axis=0)
    train_feature_exp_cols_scaled = (train_feature_exp_cols_scaled - train_feature_exp_min) / (train_feature_exp_max - train_feature_exp_min)
    valid_feature_exp_cols_scaled = (valid_feature_exp_cols_scaled - train_feature_exp_min) / (train_feature_exp_max - train_feature_exp_min)

    # Clipping valid columns to be within 0 and 1
    valid_feature_exp_cols_scaled = valid_feature_exp_cols_scaled.clip(0, 1)
    
    # Scaling Feature_clin_D_PT_age columns
    train_feature_clin_age_means = np.mean(train_feature_clin_age_cols, axis=0)
    train_feature_clin_age_stds = np.std(train_feature_clin_age_cols, axis=0)
    train_feature_clin_age_cols_scaled = (train_feature_clin_age_cols - train_feature_clin_age_means) / train_feature_clin_age_stds
    valid_feature_clin_age_cols_scaled = (valid_feature_clin_age_cols - train_feature_clin_age_means) / train_feature_clin_age_stds

    # Concatenate all scaled and non-scaled columns
    train_dataframe_scaled = pd.concat([train_noneedscale_cols, 
                                        train_feature_exp_cols_scaled, 
                                        train_feature_clin_age_cols_scaled], axis=1)
    valid_dataframe_scaled = pd.concat([valid_noneedscale_cols, 
                                        valid_feature_exp_cols_scaled, 
                                        valid_feature_clin_age_cols_scaled], axis=1)

    train_dataframe_scaled_fillna = train_dataframe_scaled.fillna(0)
    # no need fo fillna in validation dataframe because there are no NAs
    assert valid_dataframe_scaled.isna().sum().sum() == 0, "validation dataframe has NaNs"
    
    return train_dataframe_scaled_fillna, valid_dataframe_scaled