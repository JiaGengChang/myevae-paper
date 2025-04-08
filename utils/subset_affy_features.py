import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
assert load_dotenv('../.env') or load_dotenv('..env')

# subsets the training and validation dataframes 
# removes columns that are not in any of affymetrix probe sets
# HG U133 Plus 2
# Hugene 1.0 ST V1
# this is to reduce the number of NaN features in gene expression only models
# hopefully improving performance on external validation datasets
def subset_to_microarray_genes(df:pd.DataFrame) -> tuple[pd.DataFrame,list[str]]:
    
    # gene IDs with affymetrix probes
    affygenes = np.loadtxt(os.environ.get('AFFYGENESFILE'),dtype=str).tolist()
    
    # names of gene IDs used as features
    genenames = df.filter(regex='Feature_exp').columns.str.extract('^.*(ENSG.*)$')
    
    # indices of gene features that do not have affy probes
    idx_not_in_affy = np.where(~genenames.iloc[:,0].isin(affygenes))
    
    # names of gene features that do not have affy probes
    colnames_not_in_affy = df.filter(regex='Feature_exp').columns[idx_not_in_affy]
    
    # set aside genes in affy probes to return
    affy_genenames = list(set(affygenes).intersection(set(genenames.iloc[:,0])))
    
    # all features except the gene features without affy probes
    features_keep = df.columns[~df.columns.isin(colnames_not_in_affy)]
    
    # subset the feature columns
    df_keep = df[features_keep]
    
    return df_keep, affy_genenames