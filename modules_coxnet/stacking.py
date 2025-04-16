import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import argparse
from sksurv.linear_model import CoxPHSurvivalAnalysis as Model
from sksurv.util import Surv

import scipy.stats as st

import sys
sys.path.append('../utils')
from parsers import *
from parsers_external import *
from splitter import kfold_split
from scaler_external import scale_and_impute_external_dataset as scaler

parser = argparse.ArgumentParser(description='Train linear survival model. For adjusting hyperparameters, modify this file itself')
parser.add_argument('endpoint', type=str, choices=['pfs', 'os'], default='pfs', help='Survival endpoint (pfs or os)')
args = parser.parse_args()

method='std'
clin=scaler(parse_clin().set_index('PUBLIC_ID'),method).reset_index()
surv=parse_surv(args.endpoint)

rna_features=parse_rna_pc().iloc[:, :11] # use first 10 PCs
cna_features=parse_cna_pc().iloc[:, :11] # use first 10 PCs
gistic_features=parse_gistic_pc().iloc[:, :6] # use first 5/10 PCs
fish_features=parse_fish_pc().iloc[:, :11] # use first 10 PCs
sbs_features=parse_sbs() # no need PCA
sv_features=parse_sv() # no need PCA

full_dataframe = surv.merge(clin)

for features in [rna_features, cna_features]:
    full_dataframe = full_dataframe.merge(features,left_on='PUBLIC_ID',right_on='PUBLIC_ID')

full_dataframe = full_dataframe\
    .drop_duplicates()\
    .set_index('PUBLIC_ID')\
    .assign(survtime=lambda df: [max(t, 0) for t in df['survtime']])

validation_ids = parse_validation_ids(args.endpoint)

##########################

def evaluate_once(shuffle, fold):
    train_dataframe, valid_dataframe = kfold_split(full_dataframe, shuffle, fold, validation_ids)

    train_x = train_dataframe.filter(regex='^(?!survtime|survflag)')
    train_y = Surv.from_dataframe('survflag','survtime',train_dataframe)

    valid_x = valid_dataframe.filter(regex='^(?!survtime|survflag)')
    valid_y = Surv.from_dataframe('survflag','survtime',valid_dataframe)

    model=Model()
    model.fit(train_x,train_y)
    valid_metric = model.score(valid_x, valid_y)
    
    return {
            'shuffle': shuffle,
            'fold': fold,
            'valid_metric': valid_metric,
        }

results = []

for shuffle in range(10):
    for fold in range(5):
        results.append(evaluate_once(shuffle, fold))

results_df = pd.DataFrame(results)
for metric in ['valid_metric']:
    mean_metric = results_df[metric].mean()
    ci_lower, ci_upper = st.t.interval(0.95, len(results_df[metric])-1, loc=mean_metric, scale=st.sem(results_df[metric]))
    print(f"{metric} mean (95% CI): {mean_metric:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")