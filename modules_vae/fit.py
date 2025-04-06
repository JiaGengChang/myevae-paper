from torch.optim import Adam
from torch.nn import Module, MSELoss
from torch import no_grad
from torch import cat as torch_cat
from torch.utils.data import DataLoader 
from json import dump as json_dump
import sys
import os
from dotenv import load_dotenv
load_dotenv('../.env')
sys.path.append(os.environ.get("PROJECTDIR"))
from utils.validation import score_external_datasets
from utils.cindexmetric import ConcordanceIndex # metric
from utils.coxphloss import CoxPHLoss # optimization objective is negative partial log likelihood
from utils.kldivergence import KLDivergence # regularization

def fit(model:Module, trainloader:DataLoader, validloader:DataLoader, params:dict):
    """
    Trains and validates a given model using the provided data loaders and parameters.
    Args:
        model (torch.nn.Module): The model to be trained and validated.
        trainloader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        validloader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        params (Namespace): A namespace object containing training parameters such as learning rate and number of epochs.
    Returns:
        dict: A dictionary containing the training and validation history, including losses and metrics for each epoch.
    The function performs the following steps:
    1. Initializes the optimizer and loss functions.
    2. Defines the training step which includes:
        - Setting the model to training mode.
        - Zeroing the gradients.
        - Iterating over the training data to compute and backpropagate the loss.
        - Logging the training losses to the results dictionary.
    3. Defines the validation step which includes:
        - Setting the model to evaluation mode.
        - Iterating over the validation data to compute the loss without backpropagation.
        - Logging the validation losses and metrics to the results dictionary.
    4. Iterates over the specified number of epochs, calling the training and validation steps for each epoch.
    5. Returns the results dictionary containing the training and validation history.
    """
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=params.lr)
    survival_loss_func = CoxPHLoss()
    kl_loss_func = KLDivergence()
    reconstruction_loss_funcs = [MSELoss(reduction='mean') for datatype in model.input_types_vae]

    results={}
    results['params'] = {k: v for k, v in vars(params).items() if not k.startswith('_') and k != 'exp_genes'}
    results['history'] = {}

    def train_step(epoch:int):
        model.train()
        optimizer.zero_grad()
        train_reconstruction_losses = [0 for _ in range(len(model.input_types_vae))]
        train_kl_loss = 0
        train_survival_loss = 0
        for batch_idx, data in enumerate(trainloader):
            inputs_vae = [data[f'X_{input_type}'] for input_type in model.input_types_vae]
            inputs_task = [data[f'X_{input_type}'] for input_type in model.input_types_subtask]
            outputs, mu, logvar, riskpred = model.forward((inputs_vae, inputs_task))
            assert len(inputs_vae)==len(outputs)
            batch_kl_loss = params.kl_weight * kl_loss_func(mu, logvar)
            assert not batch_kl_loss.isnan().any().item()
            batch_reconstruction_losses = [
                f(output, input_vae) for f, output, input_vae in zip(reconstruction_loss_funcs, outputs, inputs_vae)
            ]
            for brl in batch_reconstruction_losses:
                assert not brl.isnan().any().item()
            batch_survival_loss = survival_loss_func(data['event_indicator'], data['event_time'], riskpred.flatten())
            assert not batch_survival_loss.isnan().any().item()
            batch_loss = batch_kl_loss + batch_survival_loss + sum(batch_reconstruction_losses)            
            batch_loss.backward()
            optimizer.step()
            train_kl_loss += batch_kl_loss.data.item()
            train_reconstruction_losses = [i + j.data.item() for i,j in zip(train_reconstruction_losses,batch_reconstruction_losses)]
            train_survival_loss += batch_survival_loss.data.item()
        
        # at the end of the epoch, log losses to results dictionary
        results['history'][epoch]['train']['kl_loss'] = train_kl_loss
        results['history'][epoch]['train']['reconstruction_loss'] = {
            input_type: loss for input_type, loss in zip(model.input_types_vae, train_reconstruction_losses)
        }
        results['history'][epoch]['train']['survival_loss'] = train_survival_loss
        return train_kl_loss, train_reconstruction_losses, train_survival_loss
    
    def valid_step(epoch:int):
        model.eval()
        event_indicator,event_time,estimate = [],[],[] # for calculating the concordance index metric
        valid_reconstruction_losses = [0 for _ in range(len(model.input_types_vae))]
        valid_kl_loss = 0
        valid_survival_loss = 0
        for batch_idx, data in enumerate(validloader):
            event_indicator.append(data['event_indicator'])
            event_time.append(data['event_time'])
            with no_grad():
                inputs_vae = [data[f'X_{input_type}'] for input_type in model.input_types_vae]
                inputs_task = [data[f'X_{input_type}'] for input_type in model.input_types_subtask]
                outputs, mu, logvar, riskpred = model.forward((inputs_vae, inputs_task))
                estimate.append(riskpred.flatten())
                assert len(inputs_vae)==len(outputs)
                batch_kl_loss = kl_loss_func(mu, logvar)
                assert not batch_kl_loss.isnan().any().item()
                batch_reconstruction_losses = [
                    f(output, input_vae) for f, output, input_vae in zip(reconstruction_loss_funcs, outputs, inputs_vae)
                ]
                for brl in batch_reconstruction_losses:
                    assert not brl.isnan().any().item()
                batch_survival_loss = survival_loss_func(data['event_indicator'], data['event_time'], riskpred.flatten())
                assert not batch_survival_loss.isnan().any().item()
            valid_kl_loss += batch_kl_loss.data.item()
            valid_reconstruction_losses = [i + j.data.item() for i,j in zip(valid_reconstruction_losses,batch_reconstruction_losses)]
            valid_survival_loss += batch_survival_loss.data.item()

        event_indicator = torch_cat(event_indicator)
        event_time = torch_cat(event_time)
        estimate = torch_cat(estimate)
        valid_metric = ConcordanceIndex(event_indicator, event_time, estimate)

        # at the end of the epoch, log losses and metrics to results dictionary
        results['history'][epoch]['valid']['kl_loss'] = valid_kl_loss
        results['history'][epoch]['valid']['reconstruction_loss'] = {
            input_type: loss for input_type, loss in zip(model.input_types_vae, valid_reconstruction_losses)
        }
        results['history'][epoch]['valid']['survival_loss'] = valid_survival_loss
        results['history'][epoch]['valid']['metric'] = valid_metric
        
        # external datasets
        # cindex_uams, cindex_hovon, cindex_emtab = score_external_datasets(model,params.endpoint)
        # results['history'][epoch]['valid']['uams_metric'] = cindex_uams
        # results['history'][epoch]['valid']['hovon_metric'] = cindex_hovon
        # results['history'][epoch]['valid']['emtab_metric'] = cindex_emtab
                
        return valid_kl_loss, valid_reconstruction_losses, valid_survival_loss, valid_metric

    # best epoch based on validation survival loss
    results['best_epoch'] = {
        'valid_survival_loss':float('inf'),
        'valid_metric': float('inf'),
        'epoch': float('inf')
    }
    # initialize early stopping variables
    patience = 20
    epochs_no_improve = 0
    best_model_state = None
    burn_in_epoch = 50 # ignore starting fluctuations in validation loss
    
    for epoch in range(params.epochs):
        if epoch not in results['history']:
            results['history'][epoch] = {'train': {}, 'valid': {}}
        train_step(epoch)
        _, _, valid_survival_loss, valid_metric = valid_step(epoch)
        
        if epoch < burn_in_epoch:
            continue
        
        # Handle early stopping
        if valid_survival_loss < results['best_epoch']['valid_survival_loss']:
            results['best_epoch']['valid_survival_loss'] = valid_survival_loss
            results['best_epoch']['valid_metric'] = valid_metric
            results['best_epoch']['epoch'] = epoch
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print('Early stopping at epoch', epoch)
            results['params']['epochs'] = epoch + 1 # record the number of epochs trained
            break
    
    # load the best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # score external datasets if using only RNASeq as input
    if params.input_types==['exp']:
        try:
            params.exp_genes # genes seen by the model. Not necessarily all input genes.
        except AttributeError:
            params.exp_genes=None
        cindex_uams, cindex_hovon, cindex_emtab = score_external_datasets(model,params.endpoint,params.shuffle,params.fold,genes=params.exp_genes)
        results['best_epoch']['uams_metric'] = cindex_uams
        results['best_epoch']['hovon_metric'] = cindex_hovon
        results['best_epoch']['emtab_metric'] = cindex_emtab

    # save results
    with open(f'{params.resultsprefix}.json', 'w') as f:
        json_dump(results, f, indent=4)