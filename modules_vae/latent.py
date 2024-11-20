import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import umap

wdir='/home/users/nus/e1083772/cancer-survival-ml/vae_models'
exptname='RNA'
shuffle='0'
fold='0'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

for ax,endpoint in zip([ax1,ax2],['pfs', 'os']):
    mu = pd.read_csv(f'{wdir}/{exptname}/{endpoint}_shuffle{shuffle}_fold{fold}.tsv',sep='\t')

    # Select the columns mu_0 to mu_127
    mu_features = mu.filter(regex='^mu_')

    # Fit and transform the data using UMAP
    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding = reducer.fit_transform(mu_features)

    ax.scatter(embedding[:, 0], embedding[:, 1], c=mu['prediction'], cmap='viridis')
    
    ax.xlabel('UMAP 1')
    ax.ylabel('UMAP 2')
    ax.colorbar(label='Risk score')
    ax.title(f'{exptname} Shuffle {shuffle} Fold {fold} {endpoint.upper()}')

plt.savefig(f'{wdir}/{exptname}/UMAP-{shuffle}-{fold}.png')