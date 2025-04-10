import pandas as pd
import numpy as np
import warnings
warnings.formatwarning = lambda msg, *args, **kwargs: f'{msg}\n'
from pycox.models import CoxPH
from sklearn.base import BaseEstimator
from sklearn.utils.validation import validate_data, check_is_fitted
from sksurv.metrics import concordance_index_censored
from torch import cat as torch_cat, no_grad, save as torch_save
from torch.nn import Module # for type checking purposes
from torch.nn.modules import activation
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch import tensor as torch_tensor

import os
from dotenv import load_dotenv
assert load_dotenv("../.env") or load_dotenv(".env")
import sys
sys.path.append(os.environ.get("PROJECTDIR"))
from utils.dataset import Dataset
from utils.buildnetwork import buildNetwork
from utils.subset_affy_features import subset_to_microarray_genes

# scikit learn compliant estimator class
# for meta-estimators like Pipeline and GridSearchCV
# implements `fit`, `predict`, and `score` functions
# these functions accept dataframe as input
# `score` returns a single np.float
class DeepSurv(BaseEstimator):
    def __init__(self,
                 input_types_all:list[str]=None,
                 subset_microarray:bool=None,
                 layer_dims:list[int]=None, 
                 activation:activation=None,
                 dropout:float=None,
                 lr:float=None, 
                 epochs:int=None,
                 burn_in:int=None,
                 patience:int=None,
                 batch_size:int=None, 
                 eventcol:str=None,
                 durationcol:str=None,
                 scale_method=None):
        self.input_types_all = input_types_all
        self.subset_microarray = subset_microarray
        self.scale_method = scale_method # scale_method is accessed but not used directly
        self.layer_dims = layer_dims 
        self.activation=activation
        self.dropout=dropout
        self.lr=lr
        self.epochs=epochs
        self.burn_in=burn_in
        self.patience=patience
        self.batch_size=batch_size
        self.eventcol = eventcol
        self.durationcol = durationcol
    
    def fit(self,X:pd.DataFrame,y=None,verbose=False):
        """
        X: dataframe with named columns according to the input type. Also contains event and duration columns.
        y: ignored, because the event and duration columns should be in X.
        verbose: whether to print loss at every epoch
        """
        assert isinstance(X,pd.DataFrame)
        assert self.layer_dims[-1] == 1 # scalar hazard output
        # input validation
        X = pd.DataFrame(validate_data(self, X, y), index=X.index, columns=X.columns)
        # remove non-microarray genes if necessary
        if self.subset_microarray:
            X, genes_keep = subset_to_microarray_genes(X)
            self.genes = genes_keep
        else:
            self.genes = None
        self.X_ = X
        self.y_ = y
        
        dataset = Dataset(X,self.input_types_all,event_indicator_col=self.eventcol,event_time_col=self.durationcol)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # lazy determination of input dim
        # which is the first element of layer dims
        input_dim = sum([getattr(dataset, f'X_{input_type}').shape[1] for input_type in self.input_types_all])
        self.network_dims = [input_dim] + self.layer_dims

        # _net is a throw-array variable because it will be replaced with self.model.net
        _net = buildNetwork(self.network_dims,self.activation,add_batchNorm=True,dropout=self.dropout)
        adam_optimizer = Adam(filter(lambda p: p.requires_grad, _net.parameters()), lr=self.lr)
        self.model = CoxPH(_net, adam_optimizer)
        self.model.net.train()
        self.model.optimizer.zero_grad()
        best_loss = np.inf
        epochs_since_best = 0
        for epoch in range(1,1+self.epochs):
            current_loss = 0 # CoxPH loss summed across batches
            for batch_idx, data in enumerate(dataloader):
                inputs = torch_cat([data[f'X_{input_type}'] for input_type in self.input_types_all],axis=-1)
                riskpred = self.model.net.forward((inputs))
                assert len(inputs)==len(riskpred)
                batch_loss = self.model.loss(riskpred.flatten(), data['event_time'], data['event_indicator'])
                assert not batch_loss.isnan().item()
                batch_loss.backward()
                self.model.optimizer.step()
                current_loss += batch_loss.item()
            
            if epoch <= self.burn_in:
                continue
            elif current_loss <= best_loss:
                epochs_since_best = 0
                best_loss = current_loss
                if verbose:
                    print(f'Epoch {epoch}: {current_loss}')
            elif epochs_since_best == self.patience:
                print(f'Early stopping at epoch {epoch - self.patience - 1}')
                return self
            else:
                epochs_since_best += 1
                
        warnings.warn(f'Early stopping not triggered. patience: {self.patience}, best_loss: {best_loss}, epochs_since_best: {epochs_since_best}')
        return self
    
    def predict(self, X:pd.DataFrame)->torch_tensor:
        """
        The order of input_types_all is absolutely critical. 
        When training and evaluating models, the input types need to be in the same order.
        Input type 'clin' should always appear last in the list
        Returns hazard predictions as a tensor of floats
        """
        # check if fit has been called
        check_is_fitted(self)
        # input validation
        X = pd.DataFrame(validate_data(self, X, reset=False), index=X.index, columns=X.columns)
        
        # remove non-microarray genes if necessary
        if self.subset_microarray:
            X, _ = subset_to_microarray_genes(X)

        self.model.net.eval()
        dataloader = DataLoader(Dataset(X, self.input_types_all, event_indicator_col=self.eventcol,event_time_col=self.durationcol), batch_size=1024, shuffle=False)
        estimates = []
        for _, data in enumerate(dataloader):
            inputs = torch_cat([data[f'X_{input_type}'] for input_type in self.input_types_all],axis=-1)
            with no_grad():
                riskpred = self.model.net.forward((inputs))
                estimates.append(riskpred.flatten())
        estimates = torch_cat(estimates) # concatenate across batches 
        return estimates
    
    def score(self, X:pd.DataFrame, y=None, loss:bool=False)->float:
        """
        loss: whether to use Cox PH loss as the metric. Default is to use C-index
        """
        assert isinstance(X, pd.DataFrame)
        X = pd.DataFrame(validate_data(self, X, reset=False),index=X.index,columns=X.columns)
        estimate = self.predict(X)
        if loss:
            # negative of the loss is our score
            duration = torch_tensor(X[self.durationcol].values)
            event = torch_tensor(X[self.eventcol].values)
            metric = -self.model.loss(estimate, duration, event).item()
            # alternative is to use c-index
        else:
            estimate = estimate.numpy()
            duration = X[self.durationcol].values
            event = X[self.eventcol].values.astype(bool)
            metric = concordance_index_censored(event,duration,estimate)[0].item()

        return metric

    def save(self, pth_path):
        torch_save(self.model.net.state_dict(), pth_path)
        return