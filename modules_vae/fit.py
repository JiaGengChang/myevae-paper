from torch.optim import Adam
from torch.nn import Module
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
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=params.lr)
    survival_loss_func = CoxPHLoss()
    kl_loss_func = KLDivergence()

    results={}
    results['params'] = {k: v for k, v in vars(params).items() if not k.startswith('_') and not k.endswith('genes')}
    results['history'] = {}

    def train_step(epoch:int):
        model.train()
        optimizer.zero_grad()
        train_reconstruction_losses = [0 for _ in range(len(model.input_types_vae))]
        train_kl_loss = 0
        train_survival_loss = 0
        results['history'][epoch] = {'train':{}, 'valid':{}}

        for batch_idx, data in enumerate(trainloader):
            inputs_vae = [data[f'X_{input_type}'] for input_type in model.input_types_vae]
            inputs_task = [data[f'X_{input_type}'] for input_type in model.input_types_subtask]
            masked_inputs, original_inputs, masks, mask_stats = model.apply_mask_to_batch(inputs_vae)
            outputs, mu, logvar, riskpred = model.forward((masked_inputs, inputs_task))
            assert len(masked_inputs)==len(outputs)
            batch_kl_loss = params.kl_weight * kl_loss_func(mu, logvar)
            assert not batch_kl_loss.isnan().any().item()
            batch_reconstruction_losses = [
                model.reconstruction_loss(output, target, mask)
                for output, target, mask in zip(outputs, original_inputs, masks)
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

        results['history'][epoch]['train']['kl_loss'] = train_kl_loss
        results['history'][epoch]['train']['reconstruction_loss'] = {
            input_type: loss for input_type, loss in zip(model.input_types_vae, train_reconstruction_losses)
        }
        results['history'][epoch]['train']['masking_stats'] = mask_stats
        results['history'][epoch]['train']['survival_loss'] = train_survival_loss
        return train_kl_loss, train_reconstruction_losses, train_survival_loss

    def valid_step(epoch:int):
        model.eval()
        event_indicator,event_time,estimate = [],[],[]
        valid_reconstruction_losses = [0 for _ in range(len(model.input_types_vae))]
        valid_kl_loss = 0
        valid_survival_loss = 0
        valid_mask_stats = {}
        for batch_idx, data in enumerate(validloader):
            event_indicator.append(data['event_indicator'])
            event_time.append(data['event_time'])
            with no_grad():
                inputs_vae = [data[f'X_{input_type}'] for input_type in model.input_types_vae]
                inputs_task = [data[f'X_{input_type}'] for input_type in model.input_types_subtask]
                masked_inputs, original_inputs, masks, mask_stats = model.apply_mask_to_batch(inputs_vae)
                outputs, mu, logvar, riskpred = model.forward((masked_inputs, inputs_task))
                estimate.append(riskpred.flatten())
                assert len(masked_inputs)==len(outputs)
                batch_kl_loss = kl_loss_func(mu, logvar)
                assert not batch_kl_loss.isnan().any().item()
                batch_reconstruction_losses = [
                    model.reconstruction_loss(output, target, mask)
                    for output, target, mask in zip(outputs, original_inputs, masks)
                ]
                for brl in batch_reconstruction_losses:
                    assert not brl.isnan().any().item()
                batch_survival_loss = survival_loss_func(data['event_indicator'], data['event_time'], riskpred.flatten())
                assert not batch_survival_loss.isnan().any().item()
                valid_mask_stats = {key: {'count': valid_mask_stats.get(key, {'count': 0})['count'] + mask_stats[key]['count'],
                                           'total_features': valid_mask_stats.get(key, {'total_features': 0})['total_features'] + mask_stats[key]['total_features'],
                                           'proportion': mask_stats[key]['proportion']}
                                    for key in set(valid_mask_stats) | set(mask_stats)}
            valid_kl_loss += batch_kl_loss.data.item()
            valid_reconstruction_losses = [i + j.data.item() for i,j in zip(valid_reconstruction_losses,batch_reconstruction_losses)]
            valid_survival_loss += batch_survival_loss.data.item()

        event_indicator = torch_cat(event_indicator)
        event_time = torch_cat(event_time)
        estimate = torch_cat(estimate)
        valid_metric = ConcordanceIndex(event_indicator, event_time, estimate)

        results['history'][epoch]['valid']['kl_loss'] = valid_kl_loss
        results['history'][epoch]['valid']['reconstruction_loss'] = {
            input_type: loss for input_type, loss in zip(model.input_types_vae, valid_reconstruction_losses)
        }
        results['history'][epoch]['valid']['masking_stats'] = {
            modality: {
                'count': stats['count'],
                'total_features': stats['total_features'],
                'mask_rate': stats['count'] / max(1, stats['total_features']),
                'proportion': stats['proportion'],
            }
            for modality, stats in valid_mask_stats.items()
        }
        results['history'][epoch]['valid']['survival_loss'] = valid_survival_loss
        results['history'][epoch]['valid']['metric'] = valid_metric
        return valid_kl_loss, valid_reconstruction_losses, valid_survival_loss, valid_metric

    results['best_epoch'] = {
        'valid_survival_loss':float('inf'),
        'valid_metric': float('inf'),
        'epoch': float('inf')
    }
    patience = 20
    epochs_no_improve = 0
    best_model_state = None
    burn_in_epoch = 50

    for epoch in range(params.epochs):
        train_step(epoch)
        _, _, valid_survival_loss, valid_metric = valid_step(epoch)

        if epoch < burn_in_epoch:
            continue

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
            results['params']['epochs'] = epoch + 1
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    if params.input_types==['exp']:
        try:
            params.exp_genes
        except AttributeError:
            params.exp_genes=None
        cindex_uams, cindex_hovon, cindex_emtab, cindex_apex = score_external_datasets(model,params)
        results['best_epoch']['uams_metric'] = cindex_uams
        results['best_epoch']['hovon_metric'] = cindex_hovon
        results['best_epoch']['emtab_metric'] = cindex_emtab
        results['best_epoch']['apex_metric'] = cindex_apex

    results['params']['all_exp_genes']=None
    results['params']['genes']=None

    with open(f'{params.resultsprefix}.json', 'w') as f:
        json_dump(results, f, indent=4)