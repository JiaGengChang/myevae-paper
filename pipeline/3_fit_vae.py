import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
assert load_dotenv('../.env')
import sys
sys.path.append('../')
import argparse
import pandas as pd
from modules_vae.fit import fit
from modules_vae.model import MultiModalVAE
from modules_vae.predict import predict_to_tsv
from utils.dataset import Dataset
from utils.plotlosses import plot_results_to_pdf
from utils.params import VAEParams
from utils.lazy_input_dims import lazy_input_dims
from utils.subset_affy_features import subset_to_microarray_genes

from torch.utils.data import DataLoader

def main():
    """
    Parse the 3 arguments which we will parallelize across. 
    the actual hyperparameters to modify are in params.py
    """
    parser = argparse.ArgumentParser(description='Train VAE model on the full dataset, for SHAP. For adjusting hyperparameters, modify params.py')
    parser.add_argument('--fulldata', action='store_true')
    parser.add_argument('--subset', action='store_true')
    parser.add_argument('--endpoint', type=str, choices=['pfs', 'os'], default='pfs', help='Survival endpoint (pfs or os)')
    parser.add_argument('--shuffle', type=int, default=0, help='Random seed for shuffling the data (0-9)')
    parser.add_argument('--fold', type=int, default=0, help='Fold number for cross-validation (0-4)')
    parser.add_argument('-f', '--fulldata', action='store_true', help='Train using the full dataset without internal validation')
    parser.add_argument('-s', '--subset', action='store_true', help='Subset expression features to genes with microarray probes')
    args = parser.parse_args()

    params = VAEParams(
        endpoint=args.endpoint,
        shuffle=args.shuffle,
        fold=args.fold,
        fulldata=args.fulldata,
        subset=args.subset,
        kl_weight=1,
        batch_size=128,
        lr=1e-4,
        epochs=300,
        burn_in=50,
        patience=20,
        input_types=['exp'],
        layer_dims=[[256, 64]],
        input_types_subtask=['clin'],
        input_dims_subtask=[5],
        layer_dims_subtask=[16, 1],
        z_dim=128,
        topKgenes=20,
        model_name='exp',
        model_type='zero_impute_naive'
    )

    os.makedirs(os.path.dirname(params.resultsprefix), exist_ok=True) # prepare output directory

    # just read shuffle and fold
    splitsdir=os.environ.get("SPLITDATADIR")
    if params.fulldata:
        train_features_file=f'{splitsdir}/full_features_{args.endpoint}_processed_mut_nan.parquet'
        train_labels_file=f'{splitsdir}/full_labels.parquet'
    else:
        train_features_file=f'{splitsdir}/{params.shuffle}/{params.fold}/train_features_{args.endpoint}_processed_mut_nan.parquet'
        train_labels_file=f'{splitsdir}/{params.shuffle}/{params.fold}/train_labels.parquet'
        valid_features_file=f'{splitsdir}/{params.shuffle}/{params.fold}/valid_features_{args.endpoint}_processed_mut_nan.parquet'
        valid_labels_file=f'{splitsdir}/{params.shuffle}/{params.fold}/valid_labels.parquet'
        assert os.path.exists(valid_features_file) and os.path.exists(valid_labels_file)
        valid_features=pd.read_parquet(valid_features_file)
        valid_labels=pd.read_parquet(valid_labels_file)[[params.eventcol,params.durationcol]]
        valid_dataframe=pd.concat([valid_labels,valid_features],axis=1)
    assert os.path.exists(train_features_file) and os.path.exists(train_labels_file)
    train_features=pd.read_parquet(train_features_file)
    train_labels=pd.read_parquet(train_labels_file)[[params.eventcol,params.durationcol]]
    train_dataframe=pd.concat([train_labels,train_features],axis=1)
    if params.subset:
        train_dataframe, _ = subset_to_microarray_genes(train_dataframe)
        if not params.fulldata:
            valid_dataframe, _ = subset_to_microarray_genes(valid_dataframe)
    eventcol = f"cens{params.endpoint}"
    durationcol = f"{params.endpoint}cdy"
    train_dataframe, mutation_feature_columns = Dataset.subset_mutation_features(train_dataframe, params.topKgenes)
    if not params.fulldata:
        valid_dataframe = Dataset.filter_mutation_features(valid_dataframe, mutation_feature_columns)
    
    trainloader = DataLoader(
        Dataset(
            train_dataframe.fillna(0.0),
            params.input_types + params.input_types_subtask,
            event_indicator_col=eventcol,
            event_time_col=durationcol,
            mutation_feature_columns=mutation_feature_columns,
        ),
        batch_size=params.batch_size,
        shuffle=True,
    )
    if params.fulldata:
        validloader = None
    else:
        validloader = DataLoader(
            Dataset(
                valid_dataframe,
                params.input_types + params.input_types_subtask,
                event_indicator_col=eventcol,
                event_time_col=durationcol,
                mutation_feature_columns=mutation_feature_columns,
            ),
            batch_size=128,
            shuffle=False,
        )

    # to set params.input_dims
    params = lazy_input_dims(train_dataframe, params)
    # for exp-only models, this is required by utils/validation.py
    params.input_types_all = params.input_types + params.input_types_subtask
    params = annotate_exp_genes(train_features, params, fieldname='genes')

    model = MultiModalVAE(input_types = params.input_types,
                        input_dims = params.input_dims,
                        layer_dims = params.layer_dims,
                        input_types_subtask = params.input_types_subtask,
                        input_dims_subtask = params.input_dims_subtask,
                        layer_dims_subtask = params.layer_dims_subtask,
                        z_dim = params.z_dim,
                        topKgenes=params.topKgenes,
                        )

    # fit and save history to json
    fit(model, trainloader, validloader, params)

    # predict on validation data once more and save to tsv
    if validloader is not None:
        predict_to_tsv(model, validloader, f'{params.resultsprefix}.tsv', save_embeddings=True)

    # plot losses and metrics to pdf
    if validloader is not None:
        plot_results_to_pdf(f'{params.resultsprefix}.json',f'{params.resultsprefix}.pdf')

    # save model state dict
    model.save(f'{params.resultsprefix}.pth')

if __name__ == "__main__":
    main()