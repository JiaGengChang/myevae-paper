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

method='rank'
# TODO find out how come 315 gene PCA does not perform as well on external validation datasets. So far best method is rank-based scaling, but still not hitting 0.6.
# maybe need to redo the 996 gene PCA.
# can try manipulating mapping the percentiles of the 315 gene PCA to the original values in 996 gene PCA, then transform with PCA. Because currently all GEP raw values are between 4-8. and this is on a different scale to RNA-Seq.

clin=scaler(parse_clin().set_index('PUBLIC_ID'),method)
surv=parse_surv(args.endpoint)
features=parse_rna_pc().iloc[:, :11] # use first 10 PCs

full_dataframe = surv\
    .merge(clin,left_on='PUBLIC_ID',right_on='PUBLIC_ID')\
    .merge(features,left_on='PUBLIC_ID',right_on='PUBLIC_ID')\
    .drop_duplicates()\
    .set_index('PUBLIC_ID')\
    .assign(survtime=lambda df: [max(t, 0) for t in df['survtime']])

validation_ids = parse_validation_ids(args.endpoint)

# process external datasets
clin_uams=scaler(parse_clin_uams(), method)
features_uams=parse_exp_pc_uams().iloc[:, :10]
uams_x=pd.concat([clin_uams, features_uams],axis=1)
uams_y=parse_surv_helper("GSE24080UAMS",args.endpoint)

clin_hovon=scaler(parse_clin_hovon(), method)
features_hovon=parse_exp_pc_hovon().iloc[:, :10]
hovon_x=pd.concat([clin_hovon, features_hovon],axis=1)
hovon_y=parse_surv_helper("HOVON65",args.endpoint)

clin_emtab=scaler(parse_clin_emtab(), method)
features_emtab=parse_exp_pc_emtab().iloc[:, :10]
emtab_x=pd.concat([clin_emtab, features_emtab],axis=1)
emtab_y=parse_surv_helper("EMTAB4032",args.endpoint)

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
    
    # also score external datasets
    uams_metric = model.score(uams_x, uams_y)
    hovon_metric = model.score(hovon_x, hovon_y)
    emtab_metric = model.score(emtab_x, emtab_y)
    return {
            'shuffle': shuffle,
            'fold': fold,
            'valid_metric': valid_metric,
            'uams_metric': uams_metric,
            'hovon_metric': hovon_metric,
            'emtab_metric': emtab_metric
        }

results = []

for shuffle in range(10):
    for fold in range(5):
        results.append(evaluate_once(shuffle, fold))

results_df = pd.DataFrame(results)
for metric in ['valid_metric', 'uams_metric', 'hovon_metric', 'emtab_metric']:
    mean_metric = results_df[metric].mean()
    ci_lower, ci_upper = st.t.interval(0.95, len(results_df[metric])-1, loc=mean_metric, scale=st.sem(results_df[metric]))
    print(f"{metric} mean (95% CI): {mean_metric:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")