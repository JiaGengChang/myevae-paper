import pandas as pd
from glob import glob
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

model_name="RNASeq256_z128_task133.64"
endpoint='pfs'
files=glob(f"/home/users/nus/e1083772/scratch/output/vae_models/{model_name}/{endpoint}_shuffle*_fold*.tsv")
dfs = [pd.read_csv(f,sep='\t') for f in files]
scores = pd.concat(dfs).groupby(('PUBLIC_ID')).mean()
scores = (scores - scores.min())/(scores.max() - scores.min())
targets = pd.read_csv('/home/users/nus/e1083772/cancer-survival-ml/FHR.tsv',sep='\t',index_col=0)['FHR'].dropna()

results = pd.concat([targets,scores],axis=1).dropna() # 469 patients
y_true, y_pred = results['FHR'].astype(int), results['prediction']

with PdfPages('/home/users/nus/e1083772/cancer-survival-ml/roc_pr_curves.pdf') as pdf:
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    RocCurveDisplay.from_predictions(y_true, y_pred, ax=ax[0])
    ax[0].set_title('ROC Curve')
    
    PrecisionRecallDisplay.from_predictions(y_true, y_pred, ax=ax[1])
    ax[1].set_title('Precision-Recall Curve')
    
    pdf.savefig(fig)
    plt.close(fig)