import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
load_dotenv('.env')
load_dotenv('../.env')
from sksurv.util import Surv

# if train features is already loaded, better to use 
def helper_get_training_genes(endpoint,shuffle,fold):
    # read the significant genes
    features_file=f'{os.environ.get("SPLITDATADIR")}/{shuffle}/{fold}/valid_features_{endpoint}_processed.parquet'
    features=pd.read_parquet(features_file)
    columns = features.filter(regex='Feature_exp').columns
    genes = columns.str.extract('.*Feature_exp_(ENSG.*)$').iloc[:,0].values.tolist()
    return genes

def geo_mean(iterable):
    a = np.array(iterable)
    return a.prod()**(1.0/len(a))

# conversion between ensgid, entrez gene id, and affy probe id
ref = pd.read_csv(os.environ.get("BIOMARTFILE"),index_col=0).convert_dtypes()

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
        .assign(D_Age=lambda df: df['D_Age'])
    
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

def parse_exp_helper(dotenvfilename,genes,level):
    ref = globals()['ref']
    geo_mean = globals()['geo_mean']
    assert ref is not None
    assert geo_mean is not None
    if level=="ensembl":
        try: # GSE24080UAMS or HOVON65
            df = pd.read_csv(os.environ.get(dotenvfilename),sep=',').sort_values('Accession')
        except: # EMTAB-4032
            df = pd.read_csv(os.environ.get(dotenvfilename),sep='\t').sort_values('Accession')
        # ensembl gene IDs
        # used for calculating indices
        df = df.rename(columns={'Accession':'PUBLIC_ID'}).set_index('PUBLIC_ID')
        gene_id_hits = [g for g in df.columns if g in genes] # 904 genes
        gene_id_miss = [g for g in genes if g not in gene_id_hits] # 92 genes
        df_hits = df[gene_id_hits]
        df_miss = pd.DataFrame(pd.NA,index=df.index,columns=gene_id_miss)
        df = pd.concat([df_hits,df_miss],axis=1)[genes]
        if not df.columns[0].startswith('Feature_exp'):
            df = df.rename(columns=lambda x: f'Feature_exp_{x}')
        return df
    elif level=="entrez":
        train_ref = ref[ref.ensembl_gene_id.isin(genes)][['ensembl_gene_id','entrezgene_id']].drop_duplicates()
        df = pd.read_csv(os.environ.get(dotenvfilename),index_col=0)
        df.index.name = 'entrezgene_id'
        df.reset_index(inplace=True)
        df_ensg = df.merge(train_ref,on='entrezgene_id').drop_duplicates().drop(columns=['entrezgene_id']).set_index('ensembl_gene_id').transpose()
        df_ensg_missing = pd.DataFrame(pd.NA, index=df_ensg.index, columns=[g for g in genes if g not in df_ensg.columns])
        df_ensg_full = pd.concat([df_ensg, df_ensg_missing],axis=1)[genes]
        return df_ensg_full
    elif level=="affy":
        df = pd.read_csv(os.environ.get(dotenvfilename),index_col=0)
        affychip = "affy_hugene_1_0_st_v1" if "EMTAB" in dotenvfilename else "affy_hg_u133_plus_2"
        df.index.name=affychip
        train_ref = ref[ref.ensembl_gene_id.isin(genes)][['ensembl_gene_id',affychip]].drop_duplicates()
        df_ensg = df.merge(train_ref,on=affychip).drop_duplicates().drop(columns=[affychip])
        public_ids = df_ensg.filter(regex='^(?!ensembl_gene_id)').columns
        df_ensg = df_ensg.groupby('ensembl_gene_id')[public_ids].agg(geo_mean)
        df_ensg = df_ensg.transpose()
        df_ensg.index.name='PUBLIC_ID'
        df_ensg_missing = pd.DataFrame(pd.NA, index=df_ensg.index, columns=[g for g in genes if g not in df_ensg.columns])
        df_ensg_full = pd.concat([df_ensg, df_ensg_missing],axis=1)
        return df_ensg_full
    else:
        raise Exception(f"{level} is not a supported ID system")

# for VAE RNA-Seq model
def parse_clin_uams():
    return parse_clin_helper("GSE24080UAMS")
def parse_clin_hovon():
    return parse_clin_helper("HOVON65")
def parse_clin_emtab():
    return parse_clin_helper("EMTAB4032")
def parse_clin_apex():
    """
    Returns a full NA for 264 patients in apex trial
    Since there is no clinical data as of now
    """
    parse_clin_emtab = globals()['parse_clin_emtab'] # get the column names
    df=pd.read_csv(os.environ.get("APEXCLINDATAFILE"),sep='\t')
    dfnas = parse_clin_emtab().reindex(df.SAMPLE)
    dfnas.index.name='PUBLIC_ID'
    return dfnas

def parse_exp_uams(genes,level):
    return parse_exp_helper("UAMSDATAFILE",genes,level)

def parse_exp_hovon(genes,level):
    return parse_exp_helper("HOVONDATAFILE",genes,level)

def parse_exp_emtab(genes,level):
    return parse_exp_helper("EMTABDATAFILE",genes,level)

def parse_exp_apex(genes,level):
    """
    only affy is supported for level
    """
    assert level=="affy"
    df=pd.read_csv(os.environ.get("APEXDATAFILE"),index_col=0)
    affychip = "affy_hg_u133_plus_2"
    df.index.name=affychip
    train_ref = ref[ref.ensembl_gene_id.isin(genes)][['ensembl_gene_id',affychip]].drop_duplicates()
    df_ensg = df.merge(train_ref,on=affychip).drop_duplicates().drop(columns=[affychip])
    public_ids = df_ensg.filter(regex='^(?!ensembl_gene_id)').columns
    df_ensg = df_ensg.groupby('ensembl_gene_id')[public_ids].agg(geo_mean)
    df_ensg = df_ensg.transpose()
    df_ensg.index.name='PUBLIC_ID'
    df_ensg_missing = pd.DataFrame(pd.NA, index=df_ensg.index, columns=[g for g in genes if g not in df_ensg.columns])
    df_ensg_full = pd.concat([df_ensg, df_ensg_missing],axis=1)
    return df_ensg_full

def parse_surv_uams(endpoint):
    return list(zip(*parse_surv_helper("GSE24080UAMS",endpoint)))
def parse_surv_hovon(endpoint):
    return list(zip(*parse_surv_helper("HOVON65",endpoint)))
def parse_surv_emtab(endpoint):
    return list(zip(*parse_surv_helper("EMTAB4032",endpoint)))
def parse_surv_apex(endpoint):
    """
    parse_surv_helper does not support APEX so just handle it here
    """
    df=pd.read_csv(os.environ.get("APEXCLINDATAFILE"),sep='\t')
    survarray = Surv.from_dataframe(f'{endpoint.upper()}_EVENT',endpoint.upper(),df)
    return list(zip(*survarray))

# for PCA RNA-Seq model
def parse_exp_pc_uams():
    return pd.read_csv(os.environ.get("UAMSPCGENEEXPRESSIONFILE"),sep='\t',index_col=0)
    
def parse_exp_pc_hovon():
    return pd.read_csv(os.environ.get("HOVONPCGENEEXPRESSIONFILE"),sep='\t',index_col=0)
    
def parse_exp_pc_emtab():
    return pd.read_csv(os.environ.get("EMTABPCGENEEXPRESSIONFILE"),sep='\t',index_col=0)