import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import umap

wdir='/home/users/nus/e1083772/cancer-survival-ml/vae_models'
exptname='RNA'
endpoint='pfs'
shuffle='0'
fold='0'

files = glob(f'{wdir}/{exptname}/{endpoint}_shuffle{shuffle}_fold{fold}.tsv')
plt.clf()

for f in files:
    mu = pd.read_csv(f,sep='\t')

    # Select the columns mu_0 to mu_127
    mu_features = mu[[f'mu_{i}' for i in range(128)]]

    # Fit and transform the data using UMAP
    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding = reducer.fit_transform(mu_features)

    # Add the UMAP components to the dataframe
    mu['umap_0'] = embedding[:, 0]
    mu['umap_1'] = embedding[:, 1]

    plt.scatter(mu['umap_0'], mu['umap_1'], c=mu['prediction'], cmap='viridis')
    
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.colorbar(label='Prediction')
plt.title(f'{exptname} {endpoint.upper()} Shuffle {shuffle} Fold {fold}')
plt.show()
plt.savefig(f'{wdir}/{exptname}/UMAP-{endpoint}-{shuffle}-{fold}.png')