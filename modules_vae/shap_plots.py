import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import argparse
import torch
import shap 
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dotenv import load_dotenv
assert load_dotenv('../.env')

# custom modules
import sys
sys.path.append('../')
from modules_vae.model import ShapMultiModalVAE
from utils.params import VAEParams
from utils.dataset import Dataset
from utils.parsers import parse_gene_reference
from utils.splitter import kfold_split
from utils.scaler import scale_and_impute_without_train_test_leak as scale_impute
from utils.lazy_input_dims import lazy_input_dims
from utils.type_prefixes import type_prefixes_dict as feature_name_patterns, sv_dict

def main():
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
            kl_weight=0,
            batch_size=0,
            lr=0,
            epochs=0,
            burn_in=0,
            patience=0,
            input_types=['mut'],
            layer_dims=[[8,4]],
            input_types_subtask=['clin'],
            input_dims_subtask=[5],
            layer_dims_subtask=[4, 1],
            z_dim=128,
            topKgenes=10,
            model_name='zero_impute_naive/mut10',
            model_type='shap'
        )


    # just r{ead shuffle and fold
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

    traindataset=Dataset(
        train_dataframe.fillna(0.0),
        params.input_types + params.input_types_subtask,
        event_indicator_col=eventcol,
        event_time_col=durationcol,
        mutation_feature_columns=mutation_feature_columns,
    )

    validdataset=Dataset(
        valid_dataframe,
        params.input_types + params.input_types_subtask,
        event_indicator_col=eventcol,
        event_time_col=durationcol,
        mutation_feature_columns=mutation_feature_columns,
    )

    # to set params.input_dims
    params = lazy_input_dims(train_dataframe, params)
    # for exp-only models, this is required by utils/validation.py
    params.input_types_all = params.input_types + params.input_types_subtask

    model = ShapMultiModalVAE(input_types = params.input_types,
                        input_dims = params.input_dims,
                        layer_dims = params.layer_dims,
                        input_types_subtask = params.input_types_subtask,
                        input_dims_subtask = params.input_dims_subtask,
                        layer_dims_subtask = params.layer_dims_subtask,
                        z_dim = params.z_dim,
                        topKgenes=params.topKgenes,
                        )

    # load model state dict
    model_checkpoint = torch.load(f'{params.resultsprefix}.pth')
    model.load_state_dict(model_checkpoint)

    # model = ShapWrapperModel(model)

    # data ~ list of tensors
    background_data = [ getattr(validdataset, f"X_{t}") for t in params.input_types + params.input_types_subtask]
    shap_data = [ getattr(traindataset, f"X_{t}") for t in params.input_types + params.input_types_subtask]

    explainer = shap.DeepExplainer(model, background_data)

    shap_values = explainer.shap_values(shap_data, check_additivity=False)

    # assign feature and shap dfs to global env
    for (i,t) in enumerate(params.input_types + params.input_types_subtask):
        globals()[f"X_{t}"] = shap_data[i]
        globals()[f"S_{t}"] = shap_values[i][:,:,0]

    # will be used for y-axis tick labels for individual modalities
    columns = valid_dataframe.columns.to_series()
    # convert ENSG ids to gene symbols
    gene_reference = parse_gene_reference()

    if 'mut' in params.input_types:
        mut_pattern = feature_name_patterns['mut']
        mut_columns = [c for c in columns if mut_pattern in c]
        mut_gene_ids = [c.split('Feature_mut_')[-1] for c in mut_columns]
        mut_names = gene_reference.get(mut_gene_ids, mut_gene_ids).tolist()
        shap.summary_plot(S_mut, X_mut, alpha=0.5, plot_type="dot", max_display=10, feature_names=mut_names, show=False)
        plt.title('Mutation status (top 10)')
        plt.tight_layout()
        plt.savefig(f'{params.resultsprefix}_shap_mut_10.png',dpi=300)
        # now for top 20
        # plt.clf()
        # shap.summary_plot(S_mut, X_mut, alpha=0.5, plot_type="dot", max_display=20, feature_names=mut_names, show=False)
        # plt.title('Mutation status (top 20)')
        # plt.tight_layout()
        # plt.savefig(f'{params.resultsprefix}_shap_mut_20.png',dpi=300)

    return

    if 'clin' in params.input_types_subtask:
        plt.clf()
        shap.summary_plot(S_clin, X_clin, plot_type="dot", alpha=0.5, feature_names=['Age','ISS 1','ISS 2','ISS 3','sexIsMale'], show=False)
        plt.title('Clinical features')
        plt.tight_layout()
        plt.savefig(f'{params.resultsprefix}_shap_clin.png',dpi=300)

    if 'ig' in params.input_types:
        sv_columns = [c for c in columns if 'Feature_SeqWGS_' in c]
        sv_names = [c.split('Feature_SeqWGS_')[-1].replace('_CALL','') for c in sv_columns]
        sv_names = [sv_dict[name] for name in sv_names]
        shap.summary_plot(S_ig, X_ig, plot_type="dot", alpha=0.5, max_display=10, feature_names=sv_names, show=False)
        plt.title('IgH translocation partners')
        plt.tight_layout()
        plt.savefig(f'{params.resultsprefix}_shap_sv.png',dpi=300)

    if 'sbs' in params.input_types:
        sbs_pattern = feature_name_patterns['sbs'] # Feature_SBS
        sbs_columns = [c for c in columns if sbs_pattern in c] 
        sbs_names = [c.split('Feature_')[-1] for c in sbs_columns]
        shap.summary_plot(S_sbs, X_sbs, alpha=0.5, plot_type="dot", max_display=10, feature_names=sbs_names, show=False)
        plt.title('SBS Mutation signatures')
        plt.tight_layout()
        plt.savefig(f'{params.resultsprefix}_shap_sbs.png',dpi=300)

    if 'fish' in params.input_types:
        fish_pattern = feature_name_patterns['fish']
        fish_columns = [c for c in columns if fish_pattern in c]
        fish_names = [c.split('_Cp_')[-1] for c in fish_columns]
        shap.summary_plot(S_fish, X_fish, alpha=0.5, plot_type="dot", max_display=10, feature_names=fish_names, show=False)
        plt.title('WGS iFISH probes copy number status (top 10)')
        plt.tight_layout()
        plt.savefig(f'{params.resultsprefix}_shap_fish.png',dpi=300)

    if 'gistic' in params.input_types:
        gistic_pattern = feature_name_patterns['gistic']
        gistic_matches = columns.str.extract(gistic_pattern, expand=False)
        gistic_columns = columns[gistic_matches.notna()].values
        gistic_names = [c.split('Feature_CNA_')[-1] for c in gistic_columns]
        shap.summary_plot(S_gistic, X_gistic, alpha=0.5, plot_type="dot", max_display=10, feature_names=gistic_names, show=False)
        plt.title('GISTIC recurrently amplified/deleleted regions (top 10)')
        plt.tight_layout()
        plt.savefig(f'{params.resultsprefix}_shap_gistic.png',dpi=300)

    if 'cna' in params.input_types:
        cna_pattern = feature_name_patterns['cna']
        cna_columns = [c for c in columns if cna_pattern in c]
        cna_gene_ids = [c.split('Feature_CNA_')[-1] for c in cna_columns]
        cna_names = [gene_reference.get(gene_id, gene_id) for gene_id in cna_gene_ids]
        shap.summary_plot(S_cna, X_cna, alpha=0.5, plot_type="dot", max_display=10, feature_names=cna_names, show=False)
        plt.title('Gene-level copy number status (top 10)')
        plt.tight_layout()
        plt.savefig(f'{params.resultsprefix}_shap_cna_all.png',dpi=300)

    if 'exp' in params.input_types:
        # feature names are Feature_exp_[ENSEMBL_GENE_ID] in X_exp dataframe, so I need to extract the gene ID from column names,
        # load in data/reference/all-human-genes.tsv all-human-genes.tsv as a dataframe,
        # map ensembl gene IDs to gene symbols, and default to gene ID if no symbol is found
        # then store the gene symbols in a list named rna_seq names
        exp_pattern = feature_name_patterns['exp']
        exp_columns = [c for c in columns if exp_pattern in c]
        exp_gene_ids = [c.split('Feature_exp_')[-1] for c in exp_columns]
        rnaseq_names = [gene_reference.get(gene_id, gene_id) for gene_id in exp_gene_ids]
        shap.summary_plot(S_exp, X_exp, alpha=0.5, plot_type="dot", max_display=10, feature_names=rnaseq_names, show=False)
        plt.title('RNA-Seq gene expression tpm (top 50)')
        plt.tight_layout()
        plt.savefig(f'{params.resultsprefix}_shap_rnaseq_all.png',dpi=300)

if __name__ == "__main__":
    main()