import os
from dotenv import load_dotenv
load_dotenv('.env')
load_dotenv('../.env')

class VAEParams:
    """
    Class namespace to hold parameters.
    Specify the parameters for the VAE model here. 
    Instantiated by main.py.
    Unlike utils/params.py, this version of VAEParams carries the hyperparameters
    """
    def __init__(self,endpoint='pfs',shuffle=0,fold=0):
        self.endpoint = endpoint
        self.shuffle = shuffle
        self.fold = fold
        # modify the rest
        self.kl_weight = 1
        self.batch_size = 128
        self.lr = 1e-4
        self.epochs = 300
        self.scale_method = "std"
        self.input_types = ['exp','cna']
        self.layer_dims = [[256], [32]]
        self.input_types_subtask = ['clin']
        self.input_dims_subtask = [5]
        self.layer_dims_subtask = [8,1]
        self.z_dim = 128
        self.model_name = 'exp-cna-latent'
        self.architecture = 'VAE'
        # do not modify these two
        self.input_types_all = self.input_types + self.input_types_subtask
        self.resultsprefix = f'{os.environ.get("OUTPUTDIR")}/vae_models/{self.model_name}/{self.endpoint}_shuffle{self.shuffle}_fold{self.fold}'
