import pandas as pd

# update params with all RNA-Seq genes used in dataset
# these are all the genes that can possibly be used for training
def annotate_exp_genes(train_dataframe:pd.DataFrame, params:dict, fieldname:str='all_exp_genes'):
    exp_features = train_dataframe.filter(regex='Feature_exp').columns.str.extract('.*(ENSG.*)').iloc[:,0].tolist()
    assert len(exp_features)>0
    setattr(params, fieldname, exp_features)
    return params