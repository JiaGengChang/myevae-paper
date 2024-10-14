import torch

def buildNetwork(layers, activation, add_batchNorm=False, dropout=0):
    net = []
    for i in range(1, len(layers)):
        net.append(torch.nn.Linear(layers[i-1], layers[i], bias=(~add_batchNorm)))
        if add_batchNorm:
            net.append(torch.nn.BatchNorm1d(layers[i]))
        if dropout > 0:
            net.append(torch.nn.Dropout(dropout))
        # add nonlinearity
        # pass a child class of torch.nn
        net.append(activation)
    outnetwork=torch.nn.Sequential(*net)
    return outnetwork.to(torch.float64)

