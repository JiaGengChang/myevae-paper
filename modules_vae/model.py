import torch
import sys
sys.path.append('../utils')
from buildnetwork import buildNetwork 

class MultiModalVAE(torch.nn.Module):
    def __init__(self,
                 input_types = ['exp','cna','gistic','sbs','fish','ig'], # data modalities for VAE
                 input_dims = [ 996,  166, 42,  10,  24,  8], # number of input features for each data modality, for VAE
                 layer_dims = [[64], [16], [4], [2], [2],[1]], # hidden layer dimensions for VAE
                 input_types_subtask = ['apobec','cth','clin'], # data modalities for sub-task
                 input_dims_subtask = [1, 1, 4], # number of input features for each data modality, for subtask network
                 layer_dims_subtask = [12,1], # hidden layer dimensions for subtask network
                 z_dim = 16, 
                ):
        super(self.__class__, self).__init__()
        
        assert all([f in ['exp','cna','gistic','sbs','fish','ig','apobec','cth'] for f in input_types]) # these predictors go into the VAE 
        assert all([f in ['apobec','cth','clin'] for f in input_types_subtask]) # these predictors may skip the VAE
        
        self.input_dims = input_dims
        self.input_dims_subtask = input_dims_subtask
        self.input_types_vae = input_types
        self.input_types_subtask = input_types_subtask
        self.bottleneck_layer_input_dims = [] 

        for input_type, input_dim, layer_dim in zip(input_types, input_dims, layer_dims):
            setattr(self, f'encoder_{input_type}', buildNetwork([input_dim] + layer_dim, activation=torch.nn.LeakyReLU()))
            setattr(self, f'decoder_{input_type}', buildNetwork(layer_dim[::-1] + [input_dim], activation=torch.nn.LeakyReLU()))
            self.bottleneck_layer_input_dims.append(layer_dim[-1])
        
        self.encoders = [getattr(self,f'encoder_{input_type}') for input_type in self.input_types_vae]
        self.decoders = [getattr(self,f'decoder_{input_type}') for input_type in self.input_types_vae]
        
        self.joint_encoders_dim = sum(self.bottleneck_layer_input_dims)
        self.joint_encoder_mu = buildNetwork([self.joint_encoders_dim,z_dim*2,z_dim],activation=torch.nn.LeakyReLU())
        self.joint_encoder_log_sigma = buildNetwork([self.joint_encoders_dim,z_dim*2,z_dim],activation=torch.nn.LeakyReLU())
        self.joint_decoder = buildNetwork([z_dim, self.joint_encoders_dim], activation=torch.nn.LeakyReLU())
        
         # sub-task e.g. survival modelling
        self.risk_predictor = buildNetwork([z_dim + sum(input_dims_subtask)] + layer_dims_subtask, activation=torch.nn.Tanh())
        
    # subroutine
    def _reparameterize(self, mu, logvar):
        if self.training:
          std = logvar.mul(0.5).exp_()
          eps = torch.autograd.Variable(std.data.new(std.size()).normal_())
          return eps.mul(std).add_(mu)
        else:
          return mu

    # internal forward pass function
    def _forward(self, xs):
        # xs is a tuple of two things
        # 1. a list of vae input tensors and 
        # 2. a risk predictor input tensor
        x_vae_list = xs[0]
        x_survival_list = xs[1]
        # check if input sizes match
        for x_task, input_dim_task in zip(x_survival_list, self.input_dims_subtask):
            assert x_task.shape[-1] == input_dim_task, 'declared input sizes in fit/forward function do not match input sizes from dataloader'
        # forward pass through peripheral encoders
        hs = [encoder(x_vae) for encoder,x_vae in zip(self.encoders, x_vae_list)]
        # concat peripheral encodings into input for bottleneck layer
        h_cat = torch.cat(hs, dim=1)
        assert not torch.isnan(h_cat).any().item(), 'nan values present in input to central encoder, h_cat'
        # pass through bottleneck layer. 1 for mean, 1 for log(variance)
        mu = self.joint_encoder_mu(h_cat)
        logvar = self.joint_encoder_log_sigma(h_cat)
        # reparameterize to get latent embedding
        z = self._reparameterize(mu, logvar)
        assert not torch.isnan(z).any().item(), 'nan values present in z-embedding'
        # concat latent embedding with risk predictor input
        x_survival_input = torch.cat((mu, torch.cat(x_survival_list, dim=1)), dim=1)
        # get risk predictions
        riskpred = self.risk_predictor(x_survival_input)
        assert not torch.isnan(riskpred).any().item(), 'nan values present in log p hazards'
        # return latent embedding, its mu and logvar, and risk predictions
        return z, mu, logvar, riskpred
    
    def decode(self, z):
        h_cat = self.joint_decoder(z) # concatenated decoded input
        hs = torch.split(h_cat, self.bottleneck_layer_input_dims, dim=1) # split into separate decoded inputs
        return [decoder(h) for decoder,h in zip(self.decoders, hs)] # decode each input separately
    
    def forward(self,xs):
        # xs is a tuple of two things
        # first item of xs is a list of tensors, the modalities to be modelled by VAE
        # second item of xs is a tensors, the clinical+ data for survival regression
        z, mu, logvar, riskpred = self._forward(xs)
        recon_xs_list = self.decode(z)
        return recon_xs_list, mu, logvar, riskpred
    
    def save(self, outfile):
        torch.save(self.state_dict(), outfile)