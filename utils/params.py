import os
from dotenv import load_dotenv
assert load_dotenv('.env') or load_dotenv('../.env')
outputdir = os.environ.get("OUTPUTDIR")
from torch.nn import LeakyReLU, Tanh

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
    def __init__(self,
                 endpoint='pfs',
                 shuffle=0,
                 fold=0,
                 model_name='exp-cna-latent',
                 fulldata=False,
                 subset=False,
                 model_type='undefined',
                 kl_weight=1,
                 batch_size=128,
                 lr=1e-4,
                 epochs=300,
                 burn_in=50,
                 patience=20,
                 scale_method='std',
                 input_types=None,
                 layer_dims=None,
                 input_types_subtask=None,
                 input_dims_subtask=None,
                 layer_dims_subtask=None,
                 z_dim=128,
                 topKgenes=None,
                 ):
        super().__init__(model_name=model_name,
                         endpoint=endpoint,
                         shuffle=shuffle,
                         fold=fold,
                         fulldata=fulldata,
                         subset=subset)
        self.architecture = 'VAE' # DO NOT MODIFY
        self.kl_weight = kl_weight
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.burn_in = burn_in
        self.patience = patience
        self.scale_method = scale_method
        self.input_types = ['exp', 'cna', 'gistic', 'fish', 'sbs', 'ig', 'mut'] if input_types is None else input_types
        self.layer_dims = [[256, 64], [128, 32], [32, 8], [16, 4], [4], [2], [4]] if layer_dims is None else layer_dims
        self.input_types_subtask = ['clin'] if input_types_subtask is None else input_types_subtask
        self.input_dims_subtask = [5] if input_dims_subtask is None else input_dims_subtask
        self.layer_dims_subtask = [16, 1] if layer_dims_subtask is None else layer_dims_subtask
        self.z_dim = z_dim
        self.topKgenes = topKgenes
         # model is trained on full data
        if self.fulldata:
            # model is trained on subset of microarray genes
            if self.subset:
                self.resultsprefix = f'{outputdir}/vae_{model_type}/{self.model_name}_subset_full/{self.endpoint}_full'
            else:
                self.resultsprefix = f'{outputdir}/vae_{model_type}/{self.model_name}_full/{self.endpoint}_full'
        # model is trained on a 80-20 split
        else:
            if self.subset:
                self.resultsprefix = f'{outputdir}/vae_{model_type}/{self.model_name}_subset/{self.endpoint}_shuffle{self.shuffle}_fold{self.fold}'
            else:
                self.resultsprefix = f'{outputdir}/vae_{model_type}/{self.model_name}/{self.endpoint}_shuffle{self.shuffle}_fold{self.fold}'
    
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