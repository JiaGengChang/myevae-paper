import pandas as pd

def annotate_exp_genes(train_dataframe:pd.DataFrame, params:dict):
    exp_features = train_dataframe.filter(regex='Feature_exp').columns.str.extract('.*(ENSG.*)').iloc[:,0].tolist()
    assert len(exp_features)>0
    params.exp_genes = exp_features
    return params