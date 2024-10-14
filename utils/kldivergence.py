import torch

import torch.nn as nn
import torch.nn.functional as F

class KLDivergence(nn.Module):
    def __init__(self):
        super(KLDivergence, self).__init__()

    def forward(self, mu, logvar):
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return kld

# Example usage:
# criterion = KLDivergence()
# loss = criterion(predictions, targets)