import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
load_dotenv('.env')
from sksurv.util import Surv

"""
Family of functions to load 3 external validation datasets:
 
parse_uams(str:endpoint) # GSE24080 (2008) University of Medicine Arkansas dataset
parse_hovon(str:endpoint) # GSE19784 (2010) GMMG4-HOVON65 German myeloma study
parse_emtab(str:endpoint) # Masaryk EMTAB-4032 study (2016)

Returns tuple of (clinical + RNA-Seq data, survival_data)
Clinical data: 
    age is min-max scaled using min and max values from CoMMpass datasets. 
    min-max scaled age of 0.5 indicates missing data (all patients in HOVON65 don't have age information)
    Gender is 1=Male and -1=Female, 0=no information
    
RNA-Seq data is min-max scaled using min and max values from the dataset itself
survival data is in the form of a structered array

Example usage:

uams_x, uams_y = parse_uams('pfs')
uams_x, uams_y = parse_uams('os')

hovon_x, hovon_y = parse_hovon('pfs')
hovon_x, hovon_y = parse_hovon('os')

emtab_x, emtab_y = parse_emtab('pfs')
emtab_x, emtab_y = parse_emtab('os')
"""

_df = pd.read_csv(os.environ.get("CLINDATAFILE"),sep='\t')
commpass_min_age = _df['D_PT_age'].min()
commpass_max_age = _df['D_PT_age'].max()
    
def scale_age_to_commpass(input_age):
    # 1: age >= max age in CoMMpass
    # 0: age <= min age in CoMMpass
    # 0.5: age unknown
    if pd.isna(input_age):
        return 0.5
    else:
        scaled_age = (input_age - commpass_min_age) / (commpass_max_age - commpass_min_age)
        scaled_age = np.clip(scaled_age, 0, 1)
        return scaled_age

def helper_get_training_genes():
    # get a list of 996 genes used for training RNA-Seq models
    df = pd.read_csv(os.environ.get("GENEEXPRESSIONFILE"),sep='\t')
    gene_ids = [s.split('Feature_exp_')[1] for s in df.filter(regex='Feature_exp').columns]
    return gene_ids

gene_ids = helper_get_training_genes()

def parse_global_clinsurv_df():
    return pd.read_csv(os.environ.get("EXTERNALCLINDATAFILE"), sep=',')\
        .sort_values(by='Patient')\
        .rename(columns={'Patient':'PUBLIC_ID'})\
        .set_index('PUBLIC_ID')

global_clinsurv_df = parse_global_clinsurv_df()

def parse_global_clindf():
    df = global_clinsurv_df\
        [['Study','D_Age','D_Gender','D_ISS']]\
        .convert_dtypes('D_ISS',int)\
        .assign(D_Age=lambda df: df['D_Age'].apply(scale_age_to_commpass))
    
    df = pd.get_dummies(df,columns=['D_ISS'],dtype=int)\
        .assign(D_male=lambda df: df['D_Gender'].map({'Male': 1, 'Female': -1}).fillna(0).astype(int))\
        .drop(columns='D_Gender')

    df.columns = df.columns\
        .str.replace('D_','Feature_clin_D_PT_')\
        .str.replace('ISS','iss') # consistency with CoMMpass clinical column names
    '''
    PUBLIC_ID [index]
    Feature_clin_D_PT_age
    Feature_clin_D_PT_iss_1
    Feature_clin_D_PT_iss_2
    Feature_clin_D_PT_iss_3
    Feature_clin_D_PT_male
    '''
    return df

global_clindf=parse_global_clindf()

def parse_clin_helper(studyname):
    return global_clindf.query(f"Study==\"{studyname}\"").drop(columns='Study')

def parse_surv_helper(studyname,endpoint):
    # returns a structured array required by sksurv linear models fit function
    ENDPOINT=str.upper(endpoint)
    studydf=global_clinsurv_df.query(f"Study==\"{studyname}\"")[[f'D_{ENDPOINT}_FLAG',f'D_{ENDPOINT}']]
    return Surv.from_dataframe(f'D_{ENDPOINT}_FLAG',f'D_{ENDPOINT}',studydf)

def _parse_exp_helper_raw(dotenvfilename):
    # used for calculating indices
    try: # GSE24080UAMS or HOVON65
        df = pd.read_csv(os.environ.get(dotenvfilename),sep=',').sort_values('Accession')
    except: # EMTAB-4032
        df = pd.read_csv(os.environ.get(dotenvfilename),sep='\t').sort_values('Accession')
    return df\
        .rename(columns={'Accession':'PUBLIC_ID'})\
        .set_index('PUBLIC_ID')

def parse_exp_helper(dotenvfilename):
    try: # GSE24080UAMS or HOVON65
        df_exp_full = pd.read_csv(os.environ.get(dotenvfilename),sep=',').sort_values('Accession')
    except: # EMTAB-4032
        df_exp_full = pd.read_csv(os.environ.get(dotenvfilename),sep='\t').sort_values('Accession')
        
    df_exp_full = df_exp_full\
        .rename(columns={'Accession':'PUBLIC_ID'})\
        .set_index('PUBLIC_ID')
    
    gene_id_hits = [g for g in df_exp_full.columns if g in gene_ids] # 904 genes
    gene_id_miss = [g for g in gene_ids if g not in gene_id_hits] # 92 genes
    df_exp_hits = df_exp_full[gene_id_hits]
    df_exp_miss = pd.DataFrame(pd.NA,index=df_exp_full.index,columns=gene_id_miss)
    df_exp = pd.concat([df_exp_hits,df_exp_miss],axis=1)[gene_ids]
    df_exp_scaled = (df_exp - df_exp.min()) / (df_exp.max() - df_exp.min()) # min max scaling
    df_exp_scaled_fillna = df_exp_scaled.fillna(0.5) # median = missing
    return df_exp_scaled_fillna

# functions to export
def parse_uams(endpoint):
    target = parse_surv_helper("GSE24080UAMS",endpoint)
    clin = parse_clin_helper("GSE24080UAMS")
    exp = parse_exp_helper("UAMSDATAFILE")
    predictors = pd.concat([clin,exp],axis=1)
    return predictors, target

def parse_hovon(endpoint):
    target = parse_surv_helper("HOVON65",endpoint)
    clin = parse_clin_helper("HOVON65")
    exp = parse_exp_helper("HOVONDATAFILE")
    predictors = pd.concat([clin,exp],axis=1)
    return predictors, target

def parse_emtab(endpoint):
    target = parse_surv_helper("EMTAB4032",endpoint)
    clin = parse_clin_helper("EMTAB4032")
    exp = parse_exp_helper("EMTABDATAFILE")
    predictors = pd.concat([clin,exp],axis=1)
    return predictors, target

