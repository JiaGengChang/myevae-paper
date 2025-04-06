import os
from dotenv import load_dotenv
assert load_dotenv('../.env') or load_dotenv('.env')
import sys
sys.path.append(os.environ.get("PROJECTDIR"))
from utils.parsers import parse_all
from utils.splitter import kfold_split

def main(REDO:bool,shuffle:int,fold:int) -> None:

    # begin writing split data
    splitsdir=os.environ.get("SPLITDATADIR")
    train_features_file=f'{splitsdir}/{shuffle}/{fold}/train_features.parquet'
    valid_features_file=f'{splitsdir}/{shuffle}/{fold}/valid_features.parquet'

    train_labels_file=f'{splitsdir}/{shuffle}/{fold}/train_labels.parquet'
    valid_labels_file=f'{splitsdir}/{shuffle}/{fold}/valid_labels.parquet'

    if not os.path.isfile(train_features_file) \
        or not os.path.isfile(valid_features_file) \
            or not os.path.isfile(train_labels_file) \
                or not os.path.isfile(valid_labels_file) \
                    or REDO:
        full_dataframe = parse_all()
        train_dataframe, valid_dataframe = kfold_split(full_dataframe, shuffle, fold)

        survcols=['oscdy','censos','pfscdy','censpfs']
        train_surv = train_dataframe.loc[:,survcols]
        valid_surv = valid_dataframe.loc[:,survcols]
        train_dataframe.drop(columns=survcols,inplace=True)
        valid_dataframe.drop(columns=survcols,inplace=True)
        
        if not os.path.exists(os.path.dirname(train_features_file)):
            os.makedirs(os.path.dirname(train_features_file), exist_ok=True)
        train_dataframe.to_parquet(train_features_file,compression=None)
        train_surv.to_parquet(train_labels_file, compression=None)
        
        if not os.path.exists(os.path.dirname(valid_features_file)):
            os.makedirs(os.path.dirname(valid_features_file), exist_ok=True)
        valid_dataframe.to_parquet(valid_features_file,compression=None)
        valid_surv.to_parquet(valid_labels_file, compression=None)
    
    return

def aux()->None:
    # creates a full dataset rather than split the data
    # useful for external validation
    # shuffle and fold are ignored
    full_dataframe = parse_all()
    survcols=['oscdy','censos','pfscdy','censpfs']
    full_surv = full_dataframe.loc[:,survcols]
    full_dataframe.drop(columns=survcols,inplace=True)
    splitsdir=os.environ.get("SPLITDATADIR")
    full_features_file=f'{splitsdir}/full_features.parquet'
    full_labels_file=f'{splitsdir}/full_labels.parquet'
    if not os.path.exists(os.path.dirname(full_features_file)):
        os.makedirs(os.path.dirname(full_features_file), exist_ok=True)
    full_dataframe.to_parquet(full_features_file,compression=None)
    full_surv.to_parquet(full_labels_file, compression=None)

    return

if __name__ == "__main__":
    REDO=True # overwrite outputs
    _pbs_array_id = int(os.getenv('PBS_ARRAY_INDEX', "-1"))
    pbs_shuffle=_pbs_array_id%10
    pbs_fold=_pbs_array_id//10

    main(REDO, pbs_shuffle, pbs_fold)
    # if REDO:
    #     aux()