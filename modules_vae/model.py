import math
import torch
import sys
sys.path.append('/home/users/nus/e1083772/cancer-survival-ml/')
from utils.buildnetwork import buildNetwork, MaskAwareNetwork

class MultiModalVAE(torch.nn.Module):
    def __init__(self,
                 # data modalities for VAE
                 input_types=['exp','cna','gistic','sbs','fish','ig'],
                 # number of input features for each data modality, for VAE
                 # e.g. [ 996,  166, 42,  10,  24,  8]
                 input_dims=[None, None, None, None, None, None],
                 # hidden layer dimensions for VAE
                 # e.g. [[64], [16], [4], [2], [2],[1]]
                 layer_dims=[[64], [16], [4], [2], [2],[1]],
                 # data modalities for sub-task
                 # e.g. ['apobec','cth','clin']
                 input_types_subtask=['clin'],
                 # number of input features for each data modality, for subtask network
                 # e.g. [1, 1, 5]
                 input_dims_subtask=[5],
                 # hidden layer dimensions for subtask network
                 # e.g. [12, 1]
                 layer_dims_subtask=[12,1],
                 # bottleneck layer dimensions
                 # e.g. 16
                 z_dim=16,
                 # an instance of activation in torch.nn
                 activation=torch.nn.LeakyReLU(),
                 # an instance of activation in torch.nn
                 # activation function for the risk network
                 subtask_activation=torch.nn.Tanh(),
                 masking_proportions=None,
                 random_state=None,
                ):
        super().__init__()

        assert all([f in ['exp','cna','gistic','sbs','fish','ig','apobec','cth'] for f in input_types])
        assert all([f in ['gistic','sbs','fish','ig','apobec','cth','clin'] for f in input_types_subtask])

        self.input_dims = input_dims
        self.input_dims_subtask = input_dims_subtask
        self.input_types_vae = input_types
        self.input_types_subtask = input_types_subtask
        self.bottleneck_layer_input_dims = []
        self.activation = activation
        self.subtask_activation = subtask_activation
        self.masking_proportions = self.validate_masking_config(masking_proportions, input_types)
        self.random_state = random_state
        self.mask_generator = torch.Generator(device='cpu') if random_state is not None else None
        if self.mask_generator is not None:
            self.mask_generator.manual_seed(int(random_state))

        for input_type, input_dim, layer_dim in zip(input_types, input_dims, layer_dims):
            setattr(self, f'encoder_{input_type}', MaskAwareNetwork(input_dim, layer_dim, activation=self.activation))
            setattr(self, f'decoder_{input_type}', buildNetwork(layer_dim[::-1] + [input_dim], activation=self.activation))
            self.bottleneck_layer_input_dims.append(layer_dim[-1])

        self.encoders = [getattr(self,f'encoder_{input_type}') for input_type in self.input_types_vae]
        self.decoders = [getattr(self,f'decoder_{input_type}') for input_type in self.input_types_vae]

        self.joint_encoders_dim = sum(self.bottleneck_layer_input_dims)
        self.joint_encoder_mu = buildNetwork([self.joint_encoders_dim,z_dim*2,z_dim],activation=self.activation)
        self.joint_encoder_log_sigma = buildNetwork([self.joint_encoders_dim,z_dim*2,z_dim],activation=self.activation)
        self.joint_decoder = buildNetwork([z_dim, self.joint_encoders_dim], activation=self.activation)

        self.risk_predictor = buildNetwork([z_dim + sum(input_dims_subtask)] + layer_dims_subtask, activation=self.subtask_activation)

    @staticmethod
    def validate_masking_config(masking_proportions, input_types):
        if masking_proportions is None:
            return {}
        if not isinstance(masking_proportions, dict):
            raise TypeError('masking_proportions must be a dict keyed by modality name or None')
        if not input_types:
            raise ValueError('input_types must be non-empty when masking is enabled')
        unknown_modalities = sorted(set(masking_proportions) - set(input_types))
        if unknown_modalities:
            raise ValueError(f'Unknown modality in masking_proportions: {unknown_modalities}. Allowed: {input_types}')
        missing_modalities = [modality for modality in input_types if modality not in masking_proportions]
        if missing_modalities:
            raise ValueError(f'masking_proportions is missing entries for modalities: {missing_modalities}')
        for modality, proportion in masking_proportions.items():
            if not math.isfinite(float(proportion)):
                raise ValueError(f'Masking proportion for modality {modality} must be finite; got {proportion!r}')
            if not 0.0 <= float(proportion) <= 1.0:
                raise ValueError(f'Masking proportion for modality {modality} must be in [0, 1]; got {proportion!r}')
        return {modality: float(proportion) for modality, proportion in masking_proportions.items()}

    @staticmethod
    def reconstruction_loss(output, target, mask=None):
        output = torch.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
        target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
        if mask is None:
            return torch.mean((output - target) ** 2)
        mask = mask.to(device=output.device, dtype=output.dtype)
        if mask.numel() == 0:
            return output.new_tensor(0.0)
        if not torch.isfinite(mask).all():
            raise ValueError('Mask contains NaN/Inf values.')
        masked_error = ((output - target) ** 2) * mask
        denom = mask.sum().clamp_min(1.0)
        return masked_error.sum() / denom

    def _sample_modality_mask(self, x, proportion, generator=None):
        if x.ndim != 2:
            raise ValueError(f'Expected a 2D modality tensor with shape (batch, features); got {tuple(x.shape)}')
        if x.numel() == 0:
            return torch.zeros_like(x)
        if not math.isfinite(float(proportion)):
            raise ValueError(f'Invalid masking proportion {proportion!r}')
        if proportion <= 0.0:
            return torch.zeros_like(x)
        if proportion >= 1.0:
            return torch.ones_like(x)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        random_values = torch.rand(x.shape, dtype=x.dtype, device=x.device, generator=generator)
        return (random_values < proportion).to(dtype=x.dtype)

    def apply_mask_to_batch(self, x_vae_list, masking_proportions=None):
        if x_vae_list is None:
            raise ValueError('x_vae_list is required for per-modality feature masking')
        if not self.input_types_vae:
            raise ValueError('No active VAE modalities are configured for masking')
        if len(x_vae_list) != len(self.input_types_vae):
            raise ValueError(f'Expected {len(self.input_types_vae)} modality tensors, got {len(x_vae_list)}')

        active_masking = masking_proportions if masking_proportions is not None else self.masking_proportions
        if active_masking is None:
            active_masking = {}

        masked_inputs = []
        original_inputs = []
        masks = []
        per_modality_stats = {}

        for idx, (x, modality) in enumerate(zip(x_vae_list, self.input_types_vae)):
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            if x.dim() != 2:
                raise ValueError(f'Modality {modality} must be rank-2 with shape (batch, features); got {tuple(x.shape)}')
            expected_dim = self.input_dims[idx] if idx < len(self.input_dims) else x.shape[1]
            if x.shape[1] != expected_dim:
                raise ValueError(f'Modality {modality} has feature dimension {x.shape[1]}, expected {expected_dim}')
            if modality not in active_masking:
                mask = torch.zeros_like(x)
                proportion = 0.0
            else:
                proportion = float(active_masking[modality])
                mask = self._sample_modality_mask(x, proportion, generator=self.mask_generator)
            corrupted = x * (1.0 - mask)
            masked_inputs.append(corrupted)
            original_inputs.append(x.clone())
            masks.append(mask)
            mask_count = int(mask.sum().item())
            total = int(mask.numel())
            per_modality_stats[modality] = {
                'proportion': proportion,
                'count': mask_count,
                'total_features': total,
                'mask_rate': float(mask_count / max(1, total)),
                'masked_fraction': float(mask_count / max(1, total)),
            }

        return masked_inputs, original_inputs, masks, per_modality_stats

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
        x_vae_list = xs[0]
        x_survival_list = xs[1]
        for x_task, input_dim_task in zip(x_survival_list, self.input_dims_subtask):
            assert x_task.shape[-1] == input_dim_task, 'declared input sizes in fit/forward function do not match input sizes from dataloader'
        hs = [encoder(x_vae) for encoder,x_vae in zip(self.encoders, x_vae_list)]
        h_cat = torch.cat(hs, dim=1)
        assert not torch.isnan(h_cat).any().item(), 'nan values present in input to central encoder, h_cat'
        mu = self.joint_encoder_mu(h_cat)
        logvar = self.joint_encoder_log_sigma(h_cat)
        z = self._reparameterize(mu, logvar)
        assert not torch.isnan(z).any().item(), 'nan values present in z-embedding'
        risk_input = mu
        if x_survival_list:
            risk_input = torch.cat((mu, torch.cat(x_survival_list, dim=1)), dim=1)
        riskpred = self.risk_predictor(risk_input)
        if not torch.isfinite(riskpred).all().item():
            raise ValueError('Risk prediction contains NaN or Inf values.')
        return z, mu, logvar, riskpred

    def decode(self, z):
        h_cat = self.joint_decoder(z)
        hs = torch.split(h_cat, self.bottleneck_layer_input_dims, dim=1)
        return [decoder(h) for decoder,h in zip(self.decoders, hs)]

    def forward(self,xs):
        z, mu, logvar, riskpred = self._forward(xs)
        recon_xs_list = self.decode(z)
        return recon_xs_list, mu, logvar, riskpred

    def save(self, outfile):
        torch.save(self.state_dict(), outfile)

class ShapMultiModalVAE(MultiModalVAE):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def __call__(self, xs):
        n_subtask = len(self.input_types_subtask)
        xs_vae = xs[:-n_subtask]
        xs_subtask = xs[-n_subtask:]
        assert isinstance(xs_vae,list)
        assert isinstance(xs_subtask,list)
        xs_rearranged = [xs_vae, xs_subtask]
        _, _, _, riskpred = self.forward(xs_rearranged)
        return riskpred