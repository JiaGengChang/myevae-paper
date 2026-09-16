import torch

def buildNetwork(layers, activation, add_batchNorm=False, dropout=0):
    net = []
    for i in range(1, len(layers)):
        net.append(torch.nn.Linear(layers[i-1], layers[i], bias=not add_batchNorm))
        if add_batchNorm:
            net.append(torch.nn.BatchNorm1d(layers[i]))
        if dropout > 0:
            net.append(torch.nn.Dropout(dropout))
        # add nonlinearity
        # pass a child class of torch.nn
        net.append(activation)
    outnetwork=torch.nn.Sequential(*net)
    return outnetwork.to(torch.float64)

class MaskAwareNetwork(torch.nn.Module):
    def __init__(self, input_dim, layers, activation, add_batchNorm=False, dropout=0):
        super().__init__()
        self.network = buildNetwork([input_dim * 2] + layers, activation, add_batchNorm, dropout)
        if layers and not add_batchNorm:
            legacy_first = torch.nn.Linear(input_dim, layers[0]).to(torch.float64)
            with torch.no_grad():
                first = self.network[0]
                first.weight[:, :input_dim].copy_(legacy_first.weight)
                first.weight[:, input_dim:].zero_()
                first.bias.copy_(legacy_first.bias)

    def forward(self, x_zero, mask):
        return self.network(torch.cat((x_zero, mask), dim=-1))

