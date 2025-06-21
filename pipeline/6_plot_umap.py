import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import umap
import argparse

wdir='/home/users/nus/e1083772/cancer-survival-ml/output/vae_models'
parser = argparse.ArgumentParser(description='UMAP of latent embeddings of a single model.')
# parser.add_argument('shuffle', type=int, help='Shuffle number')
# parser.add_argument('fold', type=int, help='Fold number')
parser.add_argument('--exptname', type=str, help='Experiment name')

args = parser.parse_args()

exptname = args.exptname
# shuffle = args.shuffle
# fold = args.fold
_pbs_array_id = int(os.getenv('PBS_ARRAY_INDEX', "-1"))
shuffle=_pbs_array_id%10
fold=_pbs_array_id//10

plt.tight_layout()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')

for ax,endpoint in zip([ax1,ax2],['os','pfs']):
    mu = pd.read_csv(f'{wdir}/{exptname}/{endpoint}_shuffle{shuffle}_fold{fold}.tsv',sep='\t')

    # Select the columns mu_0 to mu_127
    mu_features = mu.filter(regex='^mu_')

    # Fit and transform the data using UMAP
    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding = reducer.fit_transform(mu_features)

    ax_scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=mu['prediction'], cmap='viridis')
    
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    cbar = plt.colorbar(ax_scatter, ax=ax)
    cbar.set_label('Risk score')
    ax.set_title(f'{exptname} Shuffle {shuffle} Fold {fold} {endpoint.upper()}')

plt.savefig(f'{wdir}/{exptname}/UMAP-{shuffle}-{fold}.png')