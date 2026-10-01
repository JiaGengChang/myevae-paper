import pandas as pd 
from torch.utils.data import Dataset as torch_Dataset
from torch import tensor as torch_tensor, device as torch_device, float64 as torch_float64
import os
from dotenv import load_dotenv
load_dotenv(os.environ.get("PROJECTDIR"))
from utils.type_prefixes import type_prefixes_dict
import warnings

# dataset is a dataframe with columns as features (prefix Feature_) and rows as observations
# it must be a pd.Dataframe because we need its .filter method
# besides Feature_ columns, it has to have 'survflag' and 'survtime' columns which are the event indicator and event time
# survflag and survtime are targets for survival modelling
# input types is a combination of ['exp','cna','gistic','sbs','fish','ig','cth','clin']
class Dataset(torch_Dataset):
    @staticmethod
    def filter_mutation_features(df: pd.DataFrame, mutation_feature_columns):
        """Apply mutation columns selected on a training dataframe."""
        if mutation_feature_columns is None:
            return df
        mutation_columns = list(df.filter(regex='Feature_mut').columns)
        selected_set = set(mutation_feature_columns)
        columns_to_keep = [
            column for column in df.columns
            if column not in mutation_columns or column in selected_set
        ]
        return df.loc[:, columns_to_keep]

    @staticmethod
    def subset_mutation_features(df: pd.DataFrame, topKgenes=None):
        """Keep the most frequent mutation features and return their names."""
        if topKgenes is None:
            return df, None
        if not isinstance(topKgenes, int) or topKgenes < 1:
            raise ValueError("topKgenes must be a positive integer or None")

        mutation_columns = list(df.filter(regex='Feature_mut').columns)
        if not mutation_columns:
            return df, []

        # Mutation frequency is the number of non-zero, non-missing samples.
        frequencies = (
            df[mutation_columns].fillna(0).ne(0).sum(axis=0)
            .sort_values(ascending=False, kind='stable')
        )
        if len(frequencies) < topKgenes:
            warnings.warn(f"Requested top {topKgenes} mutation features, but only {len(frequencies)} available in this split.")
        selected = list(frequencies.head(min(topKgenes, len(frequencies))).index)
        selected_set = set(selected)
        columns_to_keep = [
            column for column in df.columns
            if column not in mutation_columns or column in selected_set
        ]
        return df.loc[:, columns_to_keep], selected

    def __init__(self,
                df:pd.DataFrame,
                input_types:list[str],
                event_indicator_col='survflag',
                event_time_col='survtime',
                device=torch_device("cpu"),
                offset_duration=False,
                topKgenes=None,
                mutation_feature_columns=None):
        """
        offset_duration: whether to adjust event times to non-negative numbers by min-value adjustment
        """
        if mutation_feature_columns is not None:
            df = self.filter_mutation_features(df, mutation_feature_columns)
        else:
            df, mutation_feature_columns = self.subset_mutation_features(df, topKgenes)

        self.mutation_feature_columns = mutation_feature_columns
        self.PUBLIC_ID = df.index
        self.input_types=input_types
        for input_type in input_types:
            column_prefix = type_prefixes_dict.get(input_type, None)
            if column_prefix:
                feature_values = df.filter(regex=column_prefix).values.astype(float)
                if feature_values.shape[1] == 0:
                    raise ValueError(
                        f"Input X_{input_type} has width 0: no dataframe columns match "
                        f"the expected pattern {column_prefix!r}."
                    )
                X_input = torch_tensor(feature_values, device=device).to(torch_float64)
                X_imputed = torch_tensor(pd.DataFrame(feature_values).fillna(0.0).values, device=device).to(torch_float64)
                X_mask = torch_tensor((~pd.isna(feature_values)).astype(float), device=device).to(torch_float64)
                setattr(self, f'X_{input_type}', X_input)
                setattr(self, f'X_{input_type}_imputed', X_imputed)
                setattr(self, f'X_{input_type}_mask', X_mask)
        
        self.event_indicator = df[event_indicator_col] # 0 or 1
        if offset_duration:
            # need to ensure earliest event is 0
            self.event_time = df[event_time_col] - min(0, min(df[event_time_col]))
        else:
            self.event_time = df[event_time_col]

    def __getitem__(self,index):
        # a payload with event_time, event_indicator, PUBLIC_ID, and a few tensors with prefix X_
        data = {
            'event_time': float(self.event_time.iloc[index]),
            'event_indicator': float(self.event_indicator.iloc[index]),
            'PUBLIC_ID': self.PUBLIC_ID[index]
        }
        for suffix in self.input_types:
            data[f'X_{suffix}'] = getattr(self, f'X_{suffix}', None)[index,:]
            data[f'X_{suffix}_imputed'] = getattr(self, f'X_{suffix}_imputed', None)[index,:]
            data[f'X_{suffix}_mask'] = getattr(self, f'X_{suffix}_mask', None)[index,:]
        
        return data
    
    def __len__(self):
        return len(self.PUBLIC_ID) # number of patients