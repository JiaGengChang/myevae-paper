import pandas as pd
import numpy as np

from sksurv.linear_model import CoxnetSurvivalAnalysis,CoxPHSurvivalAnalysis
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin

class TopNSelector(TransformerMixin, BaseEstimator):
    def __init__(self, n=10):
        self.n = n 
        self.features_out = None 
    
    def fit(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        # normalize the data such that each row or observation sums to 1
        X_rownormalized = X.div(X.sum(axis=1), axis=0)
        # pick the top n features with the highest sum of normalized values
        self.features_out = X_rownormalized.sum(axis=0).nlargest(self.n).index
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X.loc[:, self.features_out]
    
    def get_feature_names_out(self):
        return self.features_out

class VarianceSelector(TransformerMixin, BaseEstimator):        
    def __init__(self, **kwargs):
        self.vt = VarianceThreshold(**kwargs)
        self.features_out = None
        
    def fit(self, X, y=None, **kwargs):
        self.vt.fit(X)
        self.features_out = self.vt.get_feature_names_out()
        return self
    
    def transform(self, X):
        Xr = self.vt.transform(X)
        return pd.DataFrame(Xr, columns=self.features_out, index=X.index)

    def get_feature_names_out(self):
        return self.features_out

class CoxnetSelector(TransformerMixin, BaseEstimator):
    def __init__(self,coef_threshold=0,**kwargs):
        self.cns = CoxnetSurvivalAnalysis(**kwargs)
        self.coef_threshold = coef_threshold
        self.features_out = None
        
    def fit(self, X, y, **kwargs):
        nan = np.isnan(X).any(axis=1).values
        notnan = np.where(~nan)[0]
        self.cns.fit(X.iloc[notnan,:], y[notnan])
        _keep = np.abs(self.cns.coef_[:,-1]) > self.coef_threshold
        features_out = np.where(_keep)[0]
        self.features_out = X.columns[features_out]
        return self
    
    def transform(self, X):
        return X.loc[:,self.features_out]

    def get_feature_names_out(self):
        return np.array(self.features_out)

class CoxPHSelector(TransformerMixin, BaseEstimator):
    def __init__(self,coef_threshold=0,**kwargs):
        self.cph = CoxPHSurvivalAnalysis(**kwargs)
        self.coef_threshold = coef_threshold
        self.features_out = None
        
    def fit(self, X, y, **kwargs):
        nan = np.isnan(X).any(axis=1).values
        notnan = np.where(~nan)[0]
        self.cph.fit(X.iloc[notnan,:], y[notnan])
        _keep = np.abs(self.cph.coef_) > self.coef_threshold
        features_out = np.where(_keep)[0]
        self.features_out = X.columns[features_out]
        return self
    
    def transform(self, X):
        return X.loc[:,self.features_out]

    def get_feature_names_out(self):
        return np.array(self.features_out)

class StandardTransform(TransformerMixin, BaseEstimator):
    # scale a selection of features
    def __init__(self, cols=None):
        self.scaler = StandardScaler()
        self.cols = cols # columns to scale

    def fit(self, X, y=None):
        if self.cols is None:
            self.cols = X.columns
        self.scaler.fit(X[self.cols])
        self.feature_names_in = X.columns
        return self

    def transform(self, X):
        Xt = X.astype({col: 'float' for col in self.cols})
        Xt.loc[:,self.cols] = self.scaler.transform(Xt[self.cols])
        return Xt

    def get_feature_names_out(self):
        return np.array(self.feature_names_in)

class CorrelationSelector(TransformerMixin, BaseEstimator):
    def __init__(self, threshold=0.95):
        self.threshold = threshold
        self.features_out = None

    def fit(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        X1 = X.copy()
        corr_matrix = X.corr()
        col_corr = set() # correlated (deleted) columns
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if (abs(corr_matrix.iloc[i, j]) >= self.threshold) and (corr_matrix.columns[j] not in col_corr):
                    colname = corr_matrix.columns[i]
                    col_corr.add(colname)
                    if colname in X1.columns:
                        del X1[colname]
        self.features_out = X1.columns
        return self
    
    def transform(self, X):
        return X.loc[:, self.features_out]
    
    def get_feature_names_out(self):
        return self.features_out

class FrequencySelector(TransformerMixin, BaseEstimator):
    def __init__(self, minfreq=0.05, mincount=np.inf):
        # default is to use frequency cutoff
        self.minfreq = minfreq
        self.mincount = mincount
        self.features_out = None

    def fit(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        counts = X.sum(axis=0)
        freqs = counts / (~X.isna()).sum(axis=0)
        n_max = (~X.isna()).sum(axis=0).max()
        usecols = (counts >= self.mincount) if self.mincount/n_max < self.minfreq else freqs >= self.minfreq
        self.features_out = np.array(X.columns[usecols])
        return self

    def transform(self, X):
        return X.loc[:, self.features_out]

    def get_feature_names_out(self):
        return self.features_out

class Log1pTransform(TransformerMixin, BaseEstimator):
    def __init__(self,):
        self.features_out = None 
        
    def fit(self, X, y=None):
        self.features_out = X.columns
        return self
        
    def transform(self, X):
        logX = np.log1p(X)
        outX = pd.DataFrame(logX, index=X.index, columns=self.features_out)
        return outX 
    
    def get_feature_names_out(self):
        return self.features_out
    

class PCATransform(TransformerMixin, BaseEstimator):
    # accepts NA values unlike normal PCA
    def __init__(self, prefix=None, **kwargs):
        self.prefix = prefix if prefix else 'Unnamed_'
        self.features_out = None
        self.pca = PCA(**kwargs) # e.g. n_components
        
    def fit(self, X, y=None):
        Xfull = X.dropna()
        self.pca.fit(Xfull)
        self.features_out = np.array([self.prefix + str(s) for s in self.pca.get_feature_names_out()])
        return self
    
    def transform(self, X):
        Xfull = X.dropna()
        pcX = self.pca.transform(Xfull)
        outX = pd.DataFrame(pcX, 
                            index=Xfull.index, 
                            columns=self.features_out).reindex(X.index)
        return outX
    
    def get_feature_names_out(self):
        return self.features_out

# class OrdEncoder(TransformerMixin, BaseEstimator):
#     def __init__(self, values=[-2, -1, 0, 1, 2]):
#         self.values = values
#         self.encoder = None 
#         self.feature_names_out = None 
    
#     # encodes -2:0, -1:1, 0:2, 1:3, 2:4
#     # to comply with 0-based class label requirement XGBoost random forest classifier
#     def fit(self, X, y=None):
#         categories = [self.values for _ in range(X.shape[1])]
#         self.encoder = OrdinalEncoder(categories=categories,handle_unknown='use_encoded_value',unknown_value=np.nan).set_output(transform="pandas")
#         self.feature_names_out = X.columns
#         return self 
    
#     def transform(self, X):
#         Xout = self.encoder.fit_transform(X)
#         return Xout
        
#     def get_feature_names_out(self):
#         return self.feature_names_out