import torch

import torch.nn as nn
import torch.nn.functional as F

class KLDivergence(nn.Module):
    def __init__(self):
        super(KLDivergence, self).__init__()

    def forward(self, p, q):
        p = F.softmax(p, dim=-1)
        q = F.softmax(q, dim=-1)
        kl_div = torch.sum(p * torch.log(p / (q + 1e-10) + 1e-10), dim=-1)
        return kl_div.sum()

# Example usage:
# criterion = KLDivergence()
# loss = criterion(predictions, targets)