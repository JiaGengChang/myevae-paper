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
from splitter import kfold_split
from scaler import scale_and_impute_without_train_test_leak as scale_impute

parser = argparse.ArgumentParser(description='Train linear survival model. For adjusting hyperparameters, modify this file itself')
parser.add_argument('endpoint', type=str, choices=['pfs', 'os'], default='pfs', help='Survival endpoint (pfs or os)')
args = parser.parse_args()

clin=parse_clin()
surv=parse_surv(args.endpoint)
features = parse_rna_pc().iloc[:, :11] # use first 10 PCs

full_dataframe = surv\
    .merge(clin,left_on='PUBLIC_ID',right_on='PUBLIC_ID')\
    .merge(features,left_on='PUBLIC_ID',right_on='PUBLIC_ID')\
    .drop_duplicates()\
    .set_index('PUBLIC_ID')\
    .assign(survtime=lambda df: [max(t, 0) for t in df['survtime']])

validation_ids = parse_validation_ids(args.endpoint)
# only needed for official UAMS
# validation_ids = np.array([id for id in validation_ids if id not in ['MMRF_2788', 'MMRF_2903', 'MMRF_2905', 'MMRF_2908', 'MMRF_2914', 'MMRF_2921', 'MMRF_2924', 'MMRF_2926', 'MMRF_2938', 'MMRF_2939', 'MMRF_2940', 'MMRF_2941', 'MMRF_2946', 'MMRF_2947']])

def evaluate_once(shuffle, fold):
    train_dataframe, valid_dataframe = kfold_split(full_dataframe, shuffle, fold, validation_ids)

    train_x = train_dataframe.filter(regex='^(?!survtime|survflag)')
    train_y = Surv.from_dataframe('survflag','survtime',train_dataframe)

    valid_x = valid_dataframe.filter(regex='^(?!survtime|survflag)')
    valid_y = Surv.from_dataframe('survflag','survtime',valid_dataframe)

    model=Model()
    model.fit(train_x,train_y)
    valid_metric = model.score(valid_x, valid_y)

    return valid_metric

results = []

for shuffle in range(10):
    for fold in range(5):
        metric = evaluate_once(shuffle, fold)
        results.append({'shuffle': shuffle, 'fold': fold, 'metric': metric})

results_df = pd.DataFrame(results)

mean_metric = results_df['metric'].mean()
ci_lower, ci_upper = st.t.interval(0.95, len(results_df['metric'])-1, loc=mean_metric, scale=st.sem(results_df['metric']))

print(f"{args.endpoint} mean (95% CI): {mean_metric:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")