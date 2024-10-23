import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
load_dotenv('.env')
from sksurv.util import Surv

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
        .str.replace('Age','age')\
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

def parse_exp_helper(dotenvfilename):
    # used for calculating indices
    try: # GSE24080UAMS or HOVON65
        df = pd.read_csv(os.environ.get(dotenvfilename),sep=',').sort_values('Accession')
    except: # EMTAB-4032
        df = pd.read_csv(os.environ.get(dotenvfilename),sep='\t').sort_values('Accession')
    df = df\
        .rename(columns={'Accession':'PUBLIC_ID'})\
        .set_index('PUBLIC_ID')
    gene_id_hits = [g for g in df.columns if g in gene_ids] # 904 genes
    gene_id_miss = [g for g in gene_ids if g not in gene_id_hits] # 92 genes
    df_hits = df[gene_id_hits]
    df_miss = pd.DataFrame(pd.NA,index=df.index,columns=gene_id_miss)
    df = pd.concat([df_hits,df_miss],axis=1)[gene_ids]
    df = df.rename(columns=lambda x: f'Feature_exp_{x}')
    return df

# for VAE RNA-Seq model
def parse_clin_uams():
    return parse_clin_helper("GSE24080UAMS")
def parse_clin_hovon():
    return parse_clin_helper("HOVON65")
def parse_clin_emtab():
    return parse_clin_helper("EMTAB4032")

def parse_exp_uams():
    return parse_exp_helper("UAMSDATAFILE")
def parse_exp_hovon():
    return parse_exp_helper("HOVONDATAFILE")
def parse_exp_emtab():
    return parse_exp_helper("EMTABDATAFILE")

def parse_surv_uams(endpoint):
    return list(zip(*parse_surv_helper("GSE24080UAMS",endpoint)))
def parse_surv_hovon(endpoint):
    return list(zip(*parse_surv_helper("HOVON65",endpoint)))
def parse_surv_emtab(endpoint):
    return list(zip(*parse_surv_helper("EMTAB4032",endpoint)))

# for PCA RNA-Seq model
def parse_exp_pc_uams():
    return pd.read_csv(os.environ.get("UAMSPCGENEEXPRESSIONFILE"),sep='\t',index_col=0)
    
def parse_exp_pc_hovon():
    return pd.read_csv(os.environ.get("HOVONPCGENEEXPRESSIONFILE"),sep='\t',index_col=0)
    
def parse_exp_pc_emtab():
    return pd.read_csv(os.environ.get("EMTABPCGENEEXPRESSIONFILE"),sep='\t',index_col=0)