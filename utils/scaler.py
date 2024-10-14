import numpy as np
import pandas as pd

# scale and impute train examples
def scale_and_impute_without_train_test_leak(train_dataframe, valid_dataframe):

    train_dataframe_standardscale_cols = train_dataframe.filter(regex='^(?:Feature_exp|Feature_clin_D_PT_age)')
    train_dataframe_noneedscale_cols = train_dataframe.filter(regex='^(?!Feature_exp|Feature_clin_D_PT_age)')
    
    valid_dataframe_standardscale_cols = valid_dataframe.filter(regex='^(?:Feature_exp|Feature_clin_D_PT_age)')
    valid_dataframe_noneedscale_cols = valid_dataframe.filter(regex='^(?!Feature_exp|Feature_clin_D_PT_age)')

    train_means = np.mean(train_dataframe_standardscale_cols, axis=0)
    train_stds = np.std(train_dataframe_standardscale_cols, axis=0)
    
    train_dataframe_standardscale_cols_scaled = (train_dataframe_standardscale_cols - train_means) / train_stds
    valid_dataframe_standardscale_cols_scaled = (valid_dataframe_standardscale_cols - train_means) / train_stds
    
    train_dataframe_scaled = pd.concat([train_dataframe_noneedscale_cols, 
                                        train_dataframe_standardscale_cols_scaled], axis=1)
    valid_dataframe_scaled = pd.concat([valid_dataframe_noneedscale_cols, 
                                        valid_dataframe_standardscale_cols_scaled],axis=1)

    train_dataframe_scaled_fillna = train_dataframe_scaled.fillna(0)
    # no need fo fillna in validation dataframe because there are no NAs
    assert valid_dataframe_scaled.isna().sum().sum() == 0, "validation dataframe has NaNs"
    
    return train_dataframe_scaled_fillna, valid_dataframe_scaled