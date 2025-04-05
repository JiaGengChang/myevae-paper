import os
import argparse
import pandas as pd
from dotenv import load_dotenv
assert load_dotenv('../.env') or load_dotenv('.env')
import sys
sys.path.append(os.environ.get("PROJECTDIR"))
from modules_vae.params import VAEParams as specify_params_here
from modules_vae.fit import fit
from modules_vae.model import MultiModalVAE as Model
from modules_vae.predict import predict_to_tsv

from utils.dataset import Dataset
from utils.plotlosses import plot_results_to_pdf
from utils.subset_affy_features import subset_to_microarray_genes
from utils.lazy_input_dims import lazy_input_dims
from utils.annotate_exp_genes import annotate_exp_genes

from torch.utils.data import DataLoader

def main():
    """
    Parse the 3 arguments which we will parallelize across. 
    the actual hyperparameters to modify are in params.py
    """
    parser = argparse.ArgumentParser(description='Train VAE model. For adjusting hyperparameters, modify params.py')
    parser.add_argument('--endpoint', type=str, choices=['pfs', 'os'], default='pfs', help='Survival endpoint (pfs or os)')
    args = parser.parse_args()

    # comment out these 3 lines if not using PBS
    # PBS array ID to override shuffle and fold
    _pbs_array_id = int(os.getenv('PBS_ARRAY_INDEX', "-1"))
    pbs_shuffle=_pbs_array_id%10
    pbs_fold=_pbs_array_id//10
    params = specify_params_here(args.endpoint, pbs_shuffle, pbs_fold)

    scratchdir=os.environ.get("SPLITDATADIR")
    train_features_file=f'{scratchdir}/{params.shuffle}/{params.fold}/train_features_processed.parquet'
    valid_features_file=f'{scratchdir}/{params.shuffle}/{params.fold}/valid_features_processed.parquet'
    
    train_labels_file=f'{scratchdir}/{params.shuffle}/{params.fold}/train_labels.parquet'
    valid_labels_file=f'{scratchdir}/{params.shuffle}/{params.fold}/valid_labels.parquet'
    
    train_features=pd.read_parquet(train_features_file)
    valid_features=pd.read_parquet(valid_features_file)
    
    # subset RNA genes to those with affymetric probes
    train_features,valid_features = subset_to_microarray_genes(train_features,valid_features)

    eventcol = f"cens{params.endpoint}"
    durationcol = f"{params.endpoint}cdy"
    
    train_labels=pd.read_parquet(train_labels_file)[[eventcol,durationcol]]
    valid_labels=pd.read_parquet(valid_labels_file)[[eventcol,durationcol]]
    
    train_dataframe=pd.concat([train_labels,train_features],axis=1)
    valid_dataframe=pd.concat([valid_labels,valid_features],axis=1)
    trainloader = DataLoader(Dataset(train_dataframe, params.input_types_all, event_indicator_col=eventcol,event_time_col=durationcol), batch_size=params.batch_size, shuffle=True)
    validloader = DataLoader(Dataset(valid_dataframe, params.input_types_all, event_indicator_col=eventcol,event_time_col=durationcol), batch_size=128, shuffle=False)
    
    params = lazy_input_dims(train_dataframe, params) # determine input dimensions
    params = annotate_exp_genes(train_dataframe, params) # determine RNA genes used for training
    
    model = Model(params.input_types,
                params.input_dims,
                params.layer_dims,
                params.input_types_subtask,
                params.input_dims_subtask,
                params.layer_dims_subtask,
                params.z_dim)
    
    # create output directory
    os.makedirs(os.path.dirname(params.resultsprefix),exist_ok=True)
    
    # fit and save history to json
    fit(model, trainloader, validloader, params)

    # predict on validation data once more and save to tsv
    # predict_to_tsv(model, validloader, f'{params.resultsprefix}.tsv', save_embeddings=True)

    # plot losses and metrics to pdf
    # plot_results_to_pdf(f'{params.resultsprefix}.json',f'{params.resultsprefix}.pdf')

    # save model state dict
    # model.save(f'{params.resultsprefix}.pth')


if __name__ == "__main__":
    main()