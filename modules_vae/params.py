from dotenv import load_dotenv
load_dotenv('../.env')

class VAEParams:
    """
    Class namespace to hold parameters.
    Specify the parameters for the VAE model here. 
    Instantiated by main.py.
    """
    def __init__(self,endpoint='pfs',shuffle=0,fold=0):
        self.endpoint = endpoint
        self.shuffle = shuffle
        self.fold = fold
        # modify the rest
        self.model_name = 'example'
        self.kl_weight = 1
        self.batch_size = 128
        self.lr = 1e-4
        self.epochs = 300
        self.input_types = ['exp','cna','gistic','sbs','fish','ig']
        self.input_dims = [ 996,  166, 115,  10,  56,  8]
        self.layer_dims = [[64], [16], [8], [2], [2],[1]]
        self.input_types_subtask = ['clin']
        self.input_dims_subtask = [5]
        self.layer_dims_subtask = [4,1]
        self.z_dim = 16
        # do not modify these two
        self.input_types_all = self.input_types + self.input_types_subtask
        self.resultsprefix = f'{os.environ.get("OUTPUTDIR")}/{self.model_name}/{endpoint}_shuffle{shuffle}_fold{fold}'
    
    def __tokenize__(self):
        # a unique name for the model trained with these parameters
        # not used for now
        return '_'.join(sorted(attr+str(getattr(self, attr)).replace(" ", "") for attr in self.__dict__))
    