# this is same as pipeline/3_fit_vae.py
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
from utils.params import VAEParams
from utils.annotate_exp_genes import annotate_exp_genes
from utils.dataset import Dataset
from utils.plotlosses import plot_results_to_pdf
from utils.lazy_input_dims import lazy_input_dims

from torch.utils.data import DataLoader

def main():
    """
    Parse the 3 arguments which we will parallelize across. 
    the actual hyperparameters to modify are in params.py
    """
    parser = argparse.ArgumentParser(description='Train VAE model on the full dataset, for SHAP. For adjusting hyperparameters, modify params.py')
    parser.add_argument('--endpoint', type=str, choices=['pfs', 'os'], default='pfs', help='Survival endpoint (pfs or os)')
    parser.add_argument('--shuffle', type=int, default=0, help='Random seed for shuffling the data (0-9)')
    parser.add_argument('--fold', type=int, default=0, help='Fold number for cross-validation (0-4)')
    args = parser.parse_args()

    params = VAEParams(
        endpoint=args.endpoint,
        shuffle=args.shuffle,
        fold=args.fold,
        fulldata=False,
        subset=False,
        kl_weight=1,
        batch_size=1024,
        lr=5e-4,
        epochs=1000,
        burn_in=100,
        patience=50,
        # input_types=['exp', 'cna', 'gistic', 'fish', 'sbs', 'ig', 'mut'],
        input_types=['exp'],
        # layer_dims=[[256, 64], [128, 32], [32, 8], [16, 4], [4], [2], [4]],
        layer_dims=[[128,32]],
        input_types_subtask=['clin'],
        input_dims_subtask=[5],
        layer_dims_subtask=[4, 1],
        z_dim=128,
        topKgenes=10,
        model_name='zero_impute_naive/exp',
        model_type='shap'
    )

    os.makedirs(os.path.dirname(params.resultsprefix), exist_ok=True) # prepare output directory

    # just read shuffle and fold
    splitsdir=os.environ.get("SPLITDATADIR")
    train_features_file=f'{splitsdir}/{params.shuffle}/{params.fold}/train_features_{args.endpoint}_processed_mut_nan.parquet'
    train_labels_file=f'{splitsdir}/{params.shuffle}/{params.fold}/train_labels.parquet'
    valid_features_file=f'{splitsdir}/{params.shuffle}/{params.fold}/valid_features_{args.endpoint}_processed_mut_nan.parquet'
    valid_labels_file=f'{splitsdir}/{params.shuffle}/{params.fold}/valid_labels.parquet'
    assert os.path.exists(valid_features_file) and os.path.exists(valid_labels_file)
    valid_features=pd.read_parquet(valid_features_file)
    valid_labels=pd.read_parquet(valid_labels_file)[[params.eventcol,params.durationcol]]
    valid_dataframe=pd.concat([valid_labels,valid_features],axis=1)
    train_features=pd.read_parquet(train_features_file)
    train_labels=pd.read_parquet(train_labels_file)[[params.eventcol,params.durationcol]]
    train_dataframe=pd.concat([train_labels,train_features],axis=1)
    eventcol = f"cens{params.endpoint}"
    durationcol = f"{params.endpoint}cdy"
    train_dataframe, mutation_feature_columns = Dataset.subset_mutation_features(train_dataframe, params.topKgenes)
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
    # set genes which is required for score_external_dataset
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
    predict_to_tsv(model, validloader, f'{params.resultsprefix}.tsv', save_embeddings=True)

    # plot losses and metrics to pdf
    plot_results_to_pdf(f'{params.resultsprefix}.json',f'{params.resultsprefix}.pdf')

    # save model state dict
    model.save(f'{params.resultsprefix}.pth')

if __name__ == "__main__":
    main()