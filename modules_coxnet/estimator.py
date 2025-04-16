
import pandas as pd
import numpy as np
import warnings
warnings.formatwarning = lambda msg, *args, **kwargs: f'{msg}\n'
from sklearn.base import BaseEstimator
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sklearn.utils.validation import validate_data, check_is_fitted
from sksurv.metrics import concordance_index_censored
from torch import cat as torch_cat, tensor as torch_tensor
from sksurv.util import Surv
from json import dump as json_dump

import os
from dotenv import load_dotenv
assert load_dotenv("../.env") or load_dotenv(".env")
import sys
sys.path.append(os.environ.get("PROJECTDIR"))
from utils.dataset import Dataset
from utils.subset_affy_features import subset_to_microarray_genes

class Coxnet(BaseEstimator):
    def __init__(self,
                 eventcol=None,
                 durationcol=None,
                 input_types_all=None,
                 subset_microarray=False,
                 scale_method='std',
                 n_alphas=100, 
                 alphas=None, 
                 alpha_min_ratio='auto',
                 l1_ratio=0.5,
                 penalty_factor=None, 
                 normalize=False, 
                 copy_X=True, 
                 tol=1e-07, 
                 max_iter=100000):
        # attributes set during initialization
        self.eventcol = eventcol
        self.durationcol = durationcol
        self.subset_microarray = subset_microarray
        # scale_method is accessed but not used directly
        self.scale_method = scale_method
        # attributes to be specified or tuned by gridsearchCV
        self.input_types_all = input_types_all
        self.n_alphas = n_alphas
        self.alphas = alphas
        self.alpha_min_ratio = alpha_min_ratio
        self.l1_ratio = float(l1_ratio)
        self.penalty_factor = penalty_factor
        self.normalize = normalize
        self.copy_X = copy_X
        self.tol = float(tol)
        self.max_iter = max_iter
        
    def fit(self, X:pd.DataFrame, y=None):
        """
        X: dataframe with named columns according to the input type. Also contains event and duration columns.
        y: ignored, because duration and event columns are in X
        """
        self.model = CoxnetSurvivalAnalysis(
            n_alphas = self.n_alphas,
            alphas = self.alphas,
            alpha_min_ratio = self.alpha_min_ratio,
            l1_ratio = self.l1_ratio,
            penalty_factor = self.penalty_factor,
            normalize = self.normalize,
            copy_X = self.copy_X,
            tol = self.tol,
            max_iter = self.max_iter,
        )
        assert isinstance(X, pd.DataFrame)
        # input validation
        X = pd.DataFrame(validate_data(self, X, y), index=X.index, columns=X.columns)
        # remove non-microarray genes if necessary
        if self.subset_microarray:
            X, genes_keep = subset_to_microarray_genes(X)
            self.genes = genes_keep
        else:
            self.genes = None
        dataset = Dataset(X,self.input_types_all,event_indicator_col=self.eventcol,event_time_col=self.durationcol,offset_duration=True)
        # X_ is a torch tensor
        self.X_ = torch_cat([getattr(dataset,f"X_{t}") for t in self.input_types_all],axis=-1)
        self.y_ = Surv.from_arrays(dataset.event_indicator,dataset.event_time)
        
        self.model.fit(self.X_, self.y_)
        return self

    def predict(self, X:pd.DataFrame)->torch_tensor:
        """
        predict returns risk estimates for every sample as rows of X
        returns it as a tensor of floats
        """
        check_is_fitted(self)
        assert isinstance(X, pd.DataFrame)
        X = pd.DataFrame(validate_data(self, X, reset=False), index=X.index, columns=X.columns)
                # remove non-microarray genes if necessary
        if self.subset_microarray:
            X, _ = subset_to_microarray_genes(X)
        dataset = Dataset(X,self.input_types_all,event_indicator_col=self.eventcol,event_time_col=self.durationcol,offset_duration=True)
        X_ = torch_cat([getattr(dataset,f"X_{t}") for t in self.input_types_all],axis=-1)
        estimate = torch_tensor(self.model.predict(X_))
        return estimate

    def score(self, X:pd.DataFrame, y=None)->float:
        """
        score returns the C-index of risk estimates by the fitted Coxnet model
        a float between 0. and 1.0, usually its 0.5 or above but there is no guarantee
        y is ignored since event and time is in X
        """
        assert isinstance(X, pd.DataFrame)
        estimate = self.predict(X).numpy()
        duration = X[self.durationcol]
        event = X[self.eventcol].astype(bool)
        metric = concordance_index_censored(event,duration,estimate)[0].item()
        return metric

    def save(self,pth_path)->None:
        """
        while `pth_path` is expected to have the .pth extension
        the file is actually a json file since this scikit-surv model's params is a dictionary
        """
        with open(pth_path, 'w') as f:
            json_dump(self.model.get_params(), f, indent=4)
        

    # mimic call behaviour of torch.nn.Module
    def __call__(self, X:torch_tensor)->torch_tensor:
        return self.model.predict(X)