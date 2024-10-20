import torch

# dataset is a dataframe with columns as features (prefix Feature_) and rows as observations
# it must be a pd.Dataframe because we need its .filter method
# besides Feature_ columns, it has to have 'survflag' and 'survtime' columns which are the event indicator and event time
# survflag and survtime are targets for survival modelling
# input types is a combination of ['exp','cna','gistic','sbs','fish','ig','cth','clin']
class Dataset(torch.utils.data.Dataset):
    def __init__(self,pd_dataframe,input_types,device=torch.device("cpu"),event_indicator_col='survflag',event_time_col='survtime'):
        self.PUBLIC_ID = pd_dataframe.index
        self.input_types=input_types
        for input_type in input_types:
            column_prefix = {
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
            }.get(input_type, None)
            if column_prefix:
                X_input = torch.tensor(pd_dataframe.filter(regex=column_prefix).values.astype(float), device=device).to(torch.float64)
                setattr(self, f'X_{input_type}', X_input)
        
        self.event_indicator = pd_dataframe[event_indicator_col] # 0 or 1
        self.event_time = pd_dataframe[event_time_col] # 0+, integers

    def __getitem__(self,index):
        # a payload with event_time, event_indicator, PUBLIC_ID, and a few tensors with prefix X_
        data = {
            'event_time': self.event_time.iloc[index],
            'event_indicator': self.event_indicator.iloc[index],
            'PUBLIC_ID': self.PUBLIC_ID[index]
        }
        for suffix in self.input_types:
            data[f'X_{suffix}'] = getattr(self, f'X_{suffix}', None)[index,:]
        
        return data
    
    def __len__(self):
        return len(self.PUBLIC_ID) # number of patients