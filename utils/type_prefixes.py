# depended upon by dataset.py to extract input categories
type_prefixes_dict = {
    'apobec': 'Feature_APOBEC',
    'clin': 'Feature_clin',
    'cna': 'Feature_CNA_ENSG',
    'cth': 'Feature_chromothripsis',
    'exp': 'Feature_exp',
    'fish': 'Feature_fish',
    'gistic': 'Feature_CNA_(Amp|Del)',
    'ig': 'Feature_(RNASeq|SeqWGS)_',
    'sbs': 'Feature_SBS', 
    'emc92': 'Feature_EMC92', # gene expression signature
    'uams70': 'Feature_UAMS70', # gene expression signature
    'ifm15': 'Feature_IFM15', # gene expression signature
    'mrcix6': 'Feature_MRC_IX_6', # gene expression signature
    'exp_pca': 'Feature_exp_PC' # PCs of RNA-Seq genes
}