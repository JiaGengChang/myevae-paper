import os
import pandas as pd
import numpy as np
import warnings
warnings.formatwarning = lambda msg, *args, **kwargs: f'{msg}\n'
from torch.utils.data import DataLoader
from torch import no_grad, cat as torch_cat, tensor as torch_tensor
from torch.nn import MSELoss
from torch.optim import Adam
from sklearn.base import BaseEstimator
from sklearn.utils.validation import validate_data, check_is_fitted
from sksurv.metrics import concordance_index_censored

from dotenv import load_dotenv
assert load_dotenv('../.env') or load_dotenv('.env')
import sys
sys.path.append(os.environ.get("PROJECTDIR"))

from modules_vae.model import MultiModalVAE as Model
from utils.dataset import Dataset
from utils.coxphloss import CoxPHLoss
from utils.kldivergence import KLDivergence
from utils.subset_affy_features import subset_to_microarray_genes

# scikit learn compliant estimator class
# for meta-estimators like Pipeline and GridSearchCV
# implements `fit`, `predict`, and `score` functions
# these functions accept dataframe as input
# `score` returns a single np.float
class VAE(BaseEstimator):
    def __init__(self, 
                 input_dims:list[int], 
                 input_types:list[str]=[None], 
                 subset_microarray:bool=None,
                 layer_dims:list[list[int]]=[[None]], 
                 input_types_subtask:list[str]=None,
                 input_dims_subtask:list[int]=None, 
                 layer_dims_subtask:list[int]=None,
                 z_dim:int=None,
                 lr:float=None,
                 batch_size:int=None,
                 epochs:int=None,
                 burn_in:int=None,
                 patience:int=None, 
                 eventcol:str=None, 
                 durationcol:str=None, 
                 kl_weight:float=None,
                 scale_method:str=None):
        self.input_types = input_types
        self.subset_microarray = subset_microarray
        self.scale_method - scale_method
        self.input_dims = input_dims 
        self.layer_dims = layer_dims 
        self.input_types_subtask = input_types_subtask 
        self.input_dims_subtask = input_dims_subtask 
        self.layer_dims_subtask = layer_dims_subtask        
        self.z_dim = z_dim
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.burn_in = burn_in
        self.patience = patience
        self.eventcol = eventcol
        self.durationcol = durationcol
        self.kl_weight = kl_weight
    
    def fit(self, X:pd.DataFrame, y=None, verbose:bool=False):
        """
        X: dataframe with named columns according to the input type. Also contains event and duration columns.
        y: ignored, because the event and duration columns should be in X.
        verbose: whether to print loss at every epoch
        """
        # Check that X and y have correct shape, set n_features_in_, etc.
        X = pd.DataFrame(validate_data(self, X, y), index=X.index, columns=X.columns)
        if self.subset_microarray:
            X, genes_keep = subset_to_microarray_genes(X)
        self.genes = genes_keep
        self.X_ = X
        self.y_ = y
        assert isinstance(X,pd.DataFrame)
        # assign non-parameter attributes to estimator
        self.input_types_all = self.input_types + self.input_types_subtask
        self.model = Model(self.input_types, 
                           self.input_dims, 
                           self.layer_dims, 
                           self.input_types_subtask,
                           self.input_dims_subtask, 
                           self.layer_dims_subtask, 
                           self.z_dim)
        self.optimizer = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)
        self.survival_loss_func = CoxPHLoss()
        self.kl_loss_func = KLDivergence()
        self.reconstruction_loss_funcs = [MSELoss(reduction='mean') for datatype in self.model.input_types_vae]
        self.model.train()
        self.optimizer.zero_grad()
        trainloader = DataLoader(Dataset(X,self.input_types_all,event_indicator_col=self.eventcol,event_time_col=self.durationcol), batch_size=self.batch_size, shuffle=True)
        best_loss = np.inf
        epochs_since_best = 0
        for epoch in range(1,1+self.epochs):
            current_loss = 0 # survival loss summed across batches
            for batch_idx, data in enumerate(trainloader):
                inputs_vae = [data[f'X_{input_type}'] for input_type in self.model.input_types_vae]
                inputs_task = [data[f'X_{input_type}'] for input_type in self.model.input_types_subtask]
                outputs, mu, logvar, riskpred = self.model.forward((inputs_vae, inputs_task))
                assert len(inputs_vae)==len(outputs)
                batch_kl_loss = self.kl_weight * self.kl_loss_func(mu, logvar)
                assert not batch_kl_loss.isnan().any().item()
                batch_reconstruction_losses = [
                    f(output, input_vae) for f, output, input_vae in zip(self.reconstruction_loss_funcs, outputs, inputs_vae)
                ]
                for brl in batch_reconstruction_losses:
                    assert not brl.isnan().any().item()
                batch_survival_loss = self.survival_loss_func(data['event_indicator'], data['event_time'], riskpred.flatten())
                assert not batch_survival_loss.isnan().any().item()
                batch_loss = batch_kl_loss + batch_survival_loss + sum(batch_reconstruction_losses)
                batch_loss.backward()
                self.optimizer.step()
                current_loss += batch_loss.item()
            
            if epoch <= int(self.burn_in):
                pass
            elif current_loss < best_loss:
                epochs_since_best = 0
                best_loss = current_loss
                if verbose:
                    print(f'Epoch {epoch}: {current_loss}')
            elif epochs_since_best == int(self.patience):
                print(f'Early stopping at epoch {epoch - self.patience - 1}')
                return self
            else:
                epochs_since_best += 1
                
        warnings.warn(f'Early stopping not triggered. patience: {self.patience}, best_loss: {best_loss}, epochs_since_best: {epochs_since_best}')
        return self
        
    def predict(self, X:pd.DataFrame)->torch_tensor:
        # check if fit has been called
        check_is_fitted(self)
        if self.subset_microarray:
            X = subset_to_microarray_genes(X)
        # input validation
        X = pd.DataFrame(validate_data(self, X, reset=False), index=X.index, columns=X.columns)
        self.model.eval()
        estimates = []
        dataloader = DataLoader(Dataset(X, self.input_types_all, event_indicator_col=self.eventcol,event_time_col=self.durationcol), batch_size=1024, shuffle=False)
        for _, data in enumerate(dataloader):
            with no_grad():
                inputs_vae = [data[f'X_{input_type}'] for input_type in self.model.input_types_vae]
                inputs_task = [data[f'X_{input_type}'] for input_type in self.model.input_types_subtask]
                _, _, _, riskpred = self.model.forward((inputs_vae, inputs_task))
                estimates.append(riskpred.flatten())
        estimates = torch_cat(estimates)
        return estimates

    def score(self, X:pd.DataFrame, y=None)->float:
        assert isinstance(X, pd.DataFrame)
        X = pd.DataFrame(validate_data(self, X, reset=False),index=X.index,columns=X.columns)
        estimates = np.concatenate([t.numpy() for t in self.predict(X)])
        event = X[self.eventcol].values.astype(bool)
        duration = X[self.durationcol].values
        metric = concordance_index_censored(event, duration, estimates)[0].item()
        return metric