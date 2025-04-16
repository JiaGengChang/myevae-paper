import pandas as pd 
from torch.utils.data import Dataset as torch_Dataset
from torch import tensor as torch_tensor, device as torch_device, float64 as torch_float64
import os
from dotenv import load_dotenv
load_dotenv(os.environ.get("PROJECTDIR"))
from utils.type_prefixes import type_prefixes_dict

# dataset is a dataframe with columns as features (prefix Feature_) and rows as observations
# it must be a pd.Dataframe because we need its .filter method
# besides Feature_ columns, it has to have 'survflag' and 'survtime' columns which are the event indicator and event time
# survflag and survtime are targets for survival modelling
# input types is a combination of ['exp','cna','gistic','sbs','fish','ig','cth','clin']
class Dataset(torch_Dataset):
    def __init__(self,
                df:pd.DataFrame,
                input_types:list[str],
                event_indicator_col='survflag',
                event_time_col='survtime',
                device=torch_device("cpu"),
                offset_duration=False):
        """
        offset_duration: whether to adjust event times to non-negative numbers by min-value adjustment
        """
        self.PUBLIC_ID = df.index
        self.input_types=input_types
        for input_type in input_types:
            column_prefix = type_prefixes_dict.get(input_type, None)
            if column_prefix:
                X_input = torch_tensor(df.filter(regex=column_prefix).values.astype(float), device=device).to(torch_float64)
                setattr(self, f'X_{input_type}', X_input)
        
        self.event_indicator = df[event_indicator_col] # 0 or 1
        if offset_duration:
            # need to ensure earliest event is 0
            self.event_time = df[event_time_col] - min(0, min(df[event_time_col]))
        else:
            self.event_time = df[event_time_col]

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