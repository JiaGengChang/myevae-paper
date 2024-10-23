import pandas as pd

def scale_and_impute_external_dataset(df:pd.DataFrame, method:str=None):
    if method=='std':
        df_scaled = (df - df.mean()) / df.std()
        df_scaled_fillna = df_scaled.fillna(0.0) # 0 = missing
        
    elif method=='minmax':
        df_scaled = (df - df.min()) / (df.max() - df.min())
        df_scaled_fillna = df_scaled.fillna(0.5) # 0.5 = missing.
        
    elif method is not None:
        # scale with custom method
        try:
            df_scaled_fillna = df.apply(method)
        except:
            raise ValueError('Scaling/fillNA method \"{method}\" unknown')
    else:
        return df.fillna(0) # no scaling
    
    return df_scaled_fillna
