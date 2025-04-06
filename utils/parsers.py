import os
import pandas as pd
from dotenv import load_dotenv
assert load_dotenv('../.env') or load_dotenv('.env')

def parse_surv(): # parse cens and cdy columsn for both PFS and OS endpoints
    # surv cannot have any NA values since it is the label
    surv=pd.read_csv(os.environ.get("SURVDATAFILE"),sep='\t')\
        [['PUBLIC_ID','oscdy','censos','pfscdy','censpfs']]
    # where pfscdy / censpfs is missing, replace it with oscdy/pfscdy
    # so we get 1143 patients for both OS and PFS
    surv.loc[surv['pfscdy'].isna(),'pfscdy'] = surv.loc[surv['pfscdy'].isna(),'oscdy'] 
    surv.loc[surv['censpfs'].isna(),'censpfs'] = surv.loc[surv['censos'].isna(),'censos'] 
    surv = surv.dropna(subset=['oscdy','censos','pfscdy','censpfs'])
    surv = surv.convert_dtypes() # convert pfscdy from float64 to int64
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
    if cna.columns[0] != 'PUBLIC_ID':
        cna.reset_index(names='PUBLIC_ID',inplace=True)
    if len(cna.PUBLIC_ID[0]) > 9:
        cna.PUBLIC_ID = cna.PUBLIC_ID.str.replace('_[0-9]_BM_CD138pos','',regex=True)
    if not cna.columns[1].startswith('Feature_CNA_ENSG'):
        cna.columns = cna.columns.str.replace('ENSG', 'Feature_CNA_ENSG')
    return cna

def parse_fish():
    fish=pd.read_csv(os.environ.get("FISHFILE"),sep='\t')
    fish['Feature_fish_SeqWGS_Cp_Hyperdiploid_Call']=(fish['Feature_fish_SeqWGS_Cp_Hyperdiploid_Call']/2).astype(int)
    return fish

def parse_sv():
    sv = pd.read_csv(os.environ.get("CANONICALSVFILE"),sep='\t')
    if sv.columns[0] != 'PUBLIC_ID':
        sv.reset_index(names='PUBLIC_ID', inplace=True)
    return sv

def parse_rna():
    rna = pd.read_csv(os.environ.get("GENEEXPRESSIONFILE"),sep='\t')
    if rna.shape[0] > rna.shape[1]:
        rna = rna.set_index('Gene').T
        rna.columns.set_names(None,inplace=True)
        rna.index.set_names('PUBLIC_ID',inplace=True)
    if not rna.columns[1].startswith('Feature_exp_'):
        rna.columns = ['Feature_exp_' + c for c in rna.columns]
    if rna.columns[0] != 'PUBLIC_ID':
        rna.reset_index(names='PUBLIC_ID', inplace=True)
    if len(rna.PUBLIC_ID[0]) > 9:
        rna.PUBLIC_ID = rna.PUBLIC_ID.str.replace('_[0-9]_BM_CD138pos','',regex=True)
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
    # in the full SBS86 file, there is no PUBLIC ID column
    if sbs.columns[0]=='SAMPLE_ID':
        sbs = sbs.sort_values('SAMPLE_ID')
        sbs.SAMPLE_ID = sbs.SAMPLE_ID.str.replace('_[0-9]_BM_CD138pos','',regex=True)
        sbs = sbs.rename(columns={'SAMPLE_ID':'PUBLIC_ID'})
        sbs = sbs.groupby('PUBLIC_ID').head(n=1)
    return sbs

def parse_all():
    dfall=parse_surv()
    for parse in [parse_clin,parse_sbs,parse_cna,parse_fish,parse_rna,parse_gistic,parse_sv,parse_chromoth,parse_apobec]:
        dfnext = parse()
        if 'PUBLIC_ID' not in dfnext.columns:
            dfnext.reset_index(names='PUBLIC_ID',inplace=True)
        dfall=dfall.merge(dfnext,how='outer',left_on='PUBLIC_ID',right_on='PUBLIC_ID').drop_duplicates()
    dfall=dfall.set_index('PUBLIC_ID').dropna(subset=['oscdy','censos','pfscdy','censpfs'])
    return dfall

# subset to patients with non-NaN information in all categories
def parse_validation_ids(endpoint):
    dfvalidation = parse_all(endpoint).dropna()
    validation_ids = dfvalidation.index
    return validation_ids

# parse signarures datasets
def parse_emc92():
    df = pd.read_csv(os.environ.get("EMC92FILE"),sep='\t')
    return df

def parse_uams70():
    df = pd.read_csv(os.environ.get("UAMS70FILE"),sep='\t')
    return df

def parse_ifm15():
    df = pd.read_csv(os.environ.get("IFM15FILE"),sep='\t')
    return df

def parse_mrcix6():
    df = pd.read_csv(os.environ.get("MRCIX6FILE"),sep='\t')
    return df

# parse PCA reduced datasets
def parse_rna_pc():
    df =  pd.read_csv(os.environ.get("PCGENEEXPRESSIONFILE"),sep='\t')
    return df

def parse_cna_pc():
    df =  pd.read_csv(os.environ.get("PCCNAFILE"),sep='\t')
    return df

def parse_gistic_pc():
    df =  pd.read_csv(os.environ.get("PCGISTICFILE"),sep='\t')
    return df   

def parse_fish_pc():
    df =  pd.read_csv(os.environ.get("PCFISHFILE"),sep='\t')
    return df