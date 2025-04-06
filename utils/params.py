import os
from dotenv import load_dotenv
assert load_dotenv('.env') or load_dotenv('../.env')

class Params(None):
    """
    Base parameter class to hold parameters.
    Instantiated when specific model info is not needed, such as when loading data
    @scale_method: for transforming the microarray GEO datasets. One of 'std', 'robust', 'rank', or 'none'.
    """
    def __init__(self,endpoint='pfs',shuffle=0,fold=0,fulldata=False):
        self.endpoint = endpoint
        self.shuffle = shuffle
        self.fold = fold
        self.fulldata=fulldata
        self.durationcol = self.endpoint+'cdy'
        self.eventcol = 'cens'+self.endpoint
        self.scale_method = "std"
        self.batch_size = None
        self.lr = None 
        self.epochs = None 
        self.burn_in = None
        self.patience = None

class VAEParams(Params):
    """
    To hold additional parameters relevant for the multi-omics VAE model (myeVAE).
    Instantiated by pipeline/3_fit_gridsearchcv.py when model='VAE'
    """
    def __init__(self, **kwargs):
        super().__init__(self, **kwargs)
        self.model_name = 'exp'
        self.subset_microarray = True
        self.kl_weight = 1
        self.batch_size = 128
        self.lr = 1e-4
        self.epochs = 300
        self.burn_in = 50
        self.patience = 20
        self.scale_method = "std"
        self.input_types = ['exp']
        self.input_dims = [ None ]
        self.layer_dims = [[128]]
        self.input_types_subtask = ['clin']
        self.input_dims_subtask = [5]
        self.layer_dims_subtask = [8,1]
        self.z_dim = 16
        self.input_types_all = self.input_types + self.input_types_subtask
        if self.fulldata:
            # model is trained on full data
            self.resultsprefix = f'{os.environ.get("OUTPUTDIR")}/vae_models/{self.model_name}/{self.endpoint}_full'
        else:
            # model is trained on a 80-20 split
            self.resultsprefix = f'{os.environ.get("OUTPUTDIR")}/vae_models/{self.model_name}/{self.endpoint}_shuffle{self.shuffle}_fold{self.fold}'
        self.architecture = 'VAE'

class DeepsurvParams(Params):
    """
    To hold additional parameters relevant for the Deepsurv model.
    Instantiated by pipeline/3_fit_gridsearchcv.py when model='Deepsurv'
    """
    def __init__(self,**kwargs):
        super().__init__(self,**kwargs)
        self.model_name = 'exp'
        self.subset_microarray = True
        self.batch_size = 256
        self.lr = 1e-4
        self.epochs = 300
        self.burn_in = 50
        self.patience = 20
        self.input_types_all = ['exp','clin'] # clin should always be last
        self.layer_dims = [None, 64, 1]
        if self.fulldata:
            # model is trained on full data
            self.resultsprefix = f'{os.environ.get("OUTPUTDIR")}/deepsurv_models/{self.model_name}/{self.endpoint}_full'
        else:
            # model is trained on a 80-20 split
            self.resultsprefix = f'{os.environ.get("OUTPUTDIR")}/deepsurv_models/{self.model_name}/{self.endpoint}_shuffle{self.shuffle}_fold{self.fold}'
        self.architecture = 'Deepsurv'
