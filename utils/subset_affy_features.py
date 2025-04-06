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
def subset_to_microarray_genes(train_features:pd.DataFrame,valid_features:pd.DataFrame=None) -> tuple[pd.DataFrame]:
    
    # gene IDs with affymetrix probes
    affygenes = np.loadtxt(os.environ.get('AFFYGENESFILE'),dtype=str).tolist()
    
    # names of gene IDs used as features
    genefeatures = train_features.filter(regex='Feature_exp').columns.str.extract('^.*(ENSG.*)$')
    
    # indices of gene features that do not have affy probes
    idx_not_in_affy = np.where(~genefeatures.iloc[:,0].isin(affygenes))
    
    # names of gene features that do not have affy probes
    train_features_not_in_afffy = train_features.filter(regex='Feature_exp').columns[idx_not_in_affy]
    
    # all features except the gene features without affy probes
    features_keep = train_features.columns[~train_features.columns.isin(train_features_not_in_afffy)]
    
    # subset the feature columns
    train_features_subset = train_features[features_keep]
    
    if valid_features is not None:
        valid_features_subset = valid_features[features_keep]
        return train_features_subset, valid_features_subset
    else:
        return train_features_subset, None