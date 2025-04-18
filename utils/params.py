import os
from dotenv import load_dotenv
assert load_dotenv('.env') or load_dotenv('../.env')
outputdir = os.environ.get("OUTPUTDIR")

class Params():
    """
    Base parameter class to hold parameters.
    Instantiated when specific model info is not needed, such as when loading data
    @scale_method: for transforming the microarray GEO datasets. One of 'std', 'robust', 'rank', or 'none'.
    """
    def __init__(self,model_name:str,endpoint:str,shuffle:int,fold:int,fulldata:bool,subset:bool):
        # experiment name for the model. it will have its own output directory. usually name of the omics used.
        self.model_name = model_name
        self.endpoint = endpoint
        self.shuffle = shuffle
        self.fold = fold
        self.fulldata = fulldata
        self.subset = subset
        self.durationcol = self.endpoint+'cdy'
        self.eventcol = 'cens'+self.endpoint

class VAEParams(Params):
    """
    To hold additional parameters relevant for the multi-omics VAE model (myeVAE).
    Instantiated by pipeline/3_gridsearchcv.py when model='VAE'
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.architecture = 'VAE' # DO NOT MODIFY
        # model is trained on full data
        if self.fulldata:
            # model is trained on subset of microarray genes
            if self.subset:
                self.resultsprefix = f'{outputdir}/vae_models/{self.model_name}_subset_full/{self.endpoint}_full'
            else:
                self.resultsprefix = f'{outputdir}/vae_models/{self.model_name}_full/{self.endpoint}_full'
        # model is trained on a 80-20 split
        else:
            if self.subset:
                self.resultsprefix = f'{outputdir}/vae_models/{self.model_name}_subset/{self.endpoint}_shuffle{self.shuffle}_fold{self.fold}'
            else:
                self.resultsprefix = f'{outputdir}/vae_models/{self.model_name}/{self.endpoint}_shuffle{self.shuffle}_fold{self.fold}'
    
class DeepsurvParams(Params):
    """
    To hold additional parameters relevant for the Deepsurv model.
    Instantiated by pipeline/3_gridsearchcv.py when model='Deepsurv'
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.architecture = 'Deepsurv' # DO NOT MODIFY
        # model is trained on full data
        if self.fulldata:
            if self.subset:
                self.resultsprefix = f'{outputdir}/deepsurv_models/{self.model_name}_subset_full/{self.endpoint}_full'
            else:
                self.resultsprefix = f'{outputdir}/deepsurv_models/{self.model_name}_full/{self.endpoint}_full'
        # model is trained on a 80-20 split
        else:
            if self.subset:
                self.resultsprefix = f'{outputdir}/deepsurv_models/{self.model_name}_subset/{self.endpoint}_shuffle{self.shuffle}_fold{self.fold}'
            else:
                self.resultsprefix = f'{outputdir}/deepsurv_models/{self.model_name}/{self.endpoint}_shuffle{self.shuffle}_fold{self.fold}'

class CoxnetParams(Params):
    """
    To hold additional parameters relevant for the Elastic net Cox PH model.
    Instantiated by pipeline/3_gridsearchcv.py when model='Coxnet'
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.architecture = 'Coxnet' # DO NOT MODIFY
        # model is trained on full data
        if self.fulldata:
            if self.subset:
                self.resultsprefix = f'{outputdir}/coxnet_models/{self.model_name}_subset_full/{self.endpoint}_full'
            else:
                self.resultsprefix = f'{outputdir}/coxnet_models/{self.model_name}_full/{self.endpoint}_full'
        # model is trained on a 80-20 split
        else:
            if self.subset:
                self.resultsprefix = f'{outputdir}/coxnet_models/{self.model_name}_subset/{self.endpoint}_shuffle{self.shuffle}_fold{self.fold}'
            else:
                self.resultsprefix = f'{outputdir}/coxnet_models/{self.model_name}/{self.endpoint}_shuffle{self.shuffle}_fold{self.fold}'
                
class RSFParams(Params):
    """
    To hold additional parameters relevant for the Random survival forests model.
    Instantiated by pipeline/3_gridsearchcv.py when model='RSF'
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.architecture = 'RSF' # DO NOT MODIFY
        # model is trained on full data
        if self.fulldata:
            if self.subset:
                self.resultsprefix = f'{outputdir}/rsf_models/{self.model_name}_subset_full/{self.endpoint}_full'
            else:
                self.resultsprefix = f'{outputdir}/rsf_models/{self.model_name}_full/{self.endpoint}_full'
        # model is trained on a 80-20 split
        else:
            if self.subset:
                self.resultsprefix = f'{outputdir}/rsf_models/{self.model_name}_subset/{self.endpoint}_shuffle{self.shuffle}_fold{self.fold}'
            else:
                self.resultsprefix = f'{outputdir}/rsf_models/{self.model_name}/{self.endpoint}_shuffle{self.shuffle}_fold{self.fold}'
                
class CoxPHParams(Params):
    """
    To hold additional parameters relevant for the baseline Cox PH model.
    Instantiated by pipeline/3_gridsearchcv.py when model='CoxPH'
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs, subset=False)
        # Concept of subsetting genes not applicable to models based on risk scores
        self.architecture = 'CoxPH' # DO NOT MODIFY
        # model is trained on full data
        if self.fulldata:
            self.resultsprefix = f'{outputdir}/coxph_models/{self.model_name}_full/{self.endpoint}_full'
        # model is trained on a 80-20 split
        else:
            self.resultsprefix = f'{outputdir}/coxph_models/{self.model_name}/{self.endpoint}_shuffle{self.shuffle}_fold{self.fold}'