import os
from dotenv import load_dotenv
assert load_dotenv('.env') or load_dotenv('../.env')

class Params():
    """
    Base parameter class to hold parameters.
    Instantiated when specific model info is not needed, such as when loading data
    @scale_method: for transforming the microarray GEO datasets. One of 'std', 'robust', 'rank', or 'none'.
    """
    def __init__(self,model_name='default',endpoint='pfs',shuffle=0,fold=0,fulldata=False):
        self.model_name = model_name # determines output directory. usually name of the omics used.
        self.endpoint = endpoint
        self.shuffle = shuffle
        self.fold = fold
        self.fulldata=fulldata
        self.durationcol = self.endpoint+'cdy'
        self.eventcol = 'cens'+self.endpoint

class VAEParams(Params):
    """
    To hold additional parameters relevant for the multi-omics VAE model (myeVAE).
    Instantiated by pipeline/3_fit_gridsearchcv.py when model='VAE'
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.fulldata:
            # model is trained on full data
            self.resultsprefix = f'{os.environ.get("OUTPUTDIR")}/vae_models/{self.model_name}/{self.endpoint}_full'
        else:
            # model is trained on a 80-20 split
            self.resultsprefix = f'{os.environ.get("OUTPUTDIR")}/vae_models/{self.model_name}/{self.endpoint}_shuffle{self.shuffle}_fold{self.fold}'
        self.architecture = 'VAE' # DO NOT MODIFY
    
class DeepsurvParams(Params):
    """
    To hold additional parameters relevant for the Deepsurv model.
    Instantiated by pipeline/3_fit_gridsearchcv.py when model='Deepsurv'
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.fulldata:
            # model is trained on full data
            self.resultsprefix = f'{os.environ.get("OUTPUTDIR")}/deepsurv_models/{self.model_name}/{self.endpoint}_full'
        else:
            # model is trained on a 80-20 split
            self.resultsprefix = f'{os.environ.get("OUTPUTDIR")}/deepsurv_models/{self.model_name}/{self.endpoint}_shuffle{self.shuffle}_fold{self.fold}'
        self.architecture = 'Deepsurv' # DO NOT MODIFY