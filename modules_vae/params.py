import os
os.chdir(os.path.dirname(__file__))
from dotenv import load_dotenv
assert load_dotenv('../.env')

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
        assert os.environ.get("OUTPUTDIR") != ''
        assert os.path.exists(os.environ.get("OUTPUTDIR"))
        self.resultsprefix = f'{os.environ.get("OUTPUTDIR")}/joint_impute_vae_models/{self.model_name}/{self.endpoint}_shuffle{self.shuffle}_fold{self.fold}'
