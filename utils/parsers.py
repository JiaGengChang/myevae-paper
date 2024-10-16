import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
load_dotenv('../.env')

def parse_surv(endpoint): # pfs or os
    surv=pd.read_csv(os.environ.get("SURVDATAFILE"),sep='\t')\
        [['PUBLIC_ID',f'{endpoint}cdy',f'cens{endpoint}']]\
        .rename(columns={f'{endpoint}cdy':'survtime',f'cens{endpoint}':'survflag'})\
        .dropna()
    return surv

def parse_clin():
    clin=pd.read_csv(os.environ.get("CLINDATAFILE"),sep='\t')\
        [['PUBLIC_ID','D_PT_age','D_PT_iss','D_PT_gender']]\
        .convert_dtypes('D_PT_iss',int)
    clin=pd.get_dummies(clin,columns=['D_PT_iss','D_PT_gender'],dtype=int)
    clin = clin.drop(columns='D_PT_gender_2').rename(columns={'D_PT_gender_1':'D_PT_male'})
    clin['D_PT_male'] = clin['D_PT_male']*2 - 1 # 1 for male, -1 for female
    clin.columns = clin.columns.str.replace('D_PT_','Feature_clin_D_PT_')
    return clin

def parse_gistic():
    gistic=pd.read_csv(os.environ.get("GISTICFILE"),sep='\t')
    return gistic
    
def parse_cna():
    cna=pd.read_csv(os.environ.get("CNAFILE"),sep='\t')
    return cna

def parse_fish():
    fish=pd.read_csv(os.environ.get("FISHFILE"),sep='\t')
    fish['Feature_fish_SeqWGS_Cp_Hyperdiploid_Call']=(fish['Feature_fish_SeqWGS_Cp_Hyperdiploid_Call']/2).astype(int)
    return fish

def parse_sv():
    sv = pd.read_csv(os.environ.get("CANONICALSVFILE"),sep='\t')
    return sv

def parse_rna():
    rna = pd.read_csv(os.environ.get("GENEEXPRESSIONFILE"),sep='\t')
    return rna

def parse_chromoth():    
    chromoth = pd.read_csv(os.environ.get("CHROMOTHRIPSISFILE"),sep='\t')
    chromoth['chromothripsis'] = (chromoth['chromothripsis']=='highconf').astype(int)
    chromoth = chromoth.rename(columns={'chromothripsis':'Feature_chromothripsis'})
    return chromoth

def parse_apobec():
    apobec = pd.read_csv(os.environ.get("APOBECFILE"),sep='\t')\
        .drop(columns='APOBEC_top25')\
        .rename(columns={'APOBEC_top10':'Feature_APOBEC'})
    return apobec

def parse_sbs():
    sbs = pd.read_csv(os.environ.get("SBSFILE"), sep='\t')
    return sbs

def parse_all(endpoint):
    dfall=parse_surv(endpoint)
    for parse in [parse_clin,parse_sbs,parse_cna,parse_fish,parse_rna,parse_gistic,parse_sv,parse_chromoth,parse_apobec]:
        dfnext = parse()
        dfall=dfall.merge(dfnext,how='outer',left_on='PUBLIC_ID',right_on='PUBLIC_ID').drop_duplicates()
    dfall=dfall.set_index('PUBLIC_ID').dropna(subset=[f'survtime',f'survflag'],how='any')
    return dfall

# subset to patients with non-NaN information in all categories
def parse_validation(endpoint):
    dfvalidation = parse_all(endpoint).dropna()
    return dfvalidation