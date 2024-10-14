import torch

class CoxPHLoss(torch.nn.Module):
    def __init__(self):
        super(CoxPHLoss, self).__init__()

    def forward(self, event_indicator, event_time, outputs):

        # Sort by survival time in descending order
        order = torch.argsort(event_time, descending=True)
        outputs = outputs[order]
        event_indicator = event_indicator[order]

        # Compute the negative log partial likelihood
        neg_log_partial_likelihood = -torch.sum(event_indicator * (outputs - torch.log(torch.cumsum(torch.exp(outputs), dim=0))))
        return neg_log_partial_likelihood