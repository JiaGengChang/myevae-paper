import pandas as pd
import warnings

def scale_and_impute_external_dataset(df:pd.DataFrame, method:str='minmax'):
    # for now perform scaling on all column
    columns = df.columns
    df_scaled = df.copy()
    if method=='std':
        df_scaled[columns] = (df[columns] - df[columns].mean()) / df[columns].std()
    elif method=='minmax':
        df_scaled[columns] = (df[columns] - df[columns].min()) / (df[columns].max() - df[columns].min())
    elif method=='robust':
        df_scaled[columns] = (df[columns] - df[columns].median()) / (df[columns].quantile(0.95) - df[columns].quantile(0.05))
    elif method=='rank':
        df_scaled[columns] = df[columns].rank(method='max', pct=True)
    elif method is not None:
        try:
            # scale with custom method
            df_scaled[columns] = df[columns].apply(method)
        except:
            raise ValueError('Scaling/fillNA method \"{method}\" unknown')
    else:
        # no scaling
        warnings.warn('No scaling method specified. No scaling, only imputing NA with 0.')
        return df.fillna(0)
    
    df_scaled[columns] += 1e-5 # to distinguish actual 0 values from NA values
    
    return df_scaled.fillna(0.0).astype(float)
