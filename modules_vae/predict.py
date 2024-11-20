import torch
import numpy as np
import pandas as pd

def predict_to_tsv(model, loader, outputfile, save_embeddings=False):
    model.eval()
    predictions = []
    public_ids = []
    if save_embeddings:
        mu_list = []
    for batch_idx, data in enumerate(loader):
        with torch.no_grad():
            inputs_vae = [data[f'X_{input_type}'] for input_type in model.input_types_vae]
            inputs_task = [data[f'X_{input_type}'] for input_type in model.input_types_subtask]
            _, mu, _, riskpred = model.forward((inputs_vae, inputs_task))
            predictions.append(riskpred.detach().numpy())
            if save_embeddings:
                mu_list.append(mu.detach().numpy())
            public_ids.append(data['PUBLIC_ID'])
    predictions = np.concatenate(predictions)
    public_ids = np.concatenate(public_ids)
    if save_embeddings:
        mu_list = np.concatenate(mu_list)
    df = pd.DataFrame({
        'PUBLIC_ID': public_ids,
        'prediction': predictions.flatten()
    })
    if save_embeddings:
        mu_columns = [f'mu_{i}' for i in range(mu_list.shape[1])]
        df[mu_columns] = pd.DataFrame(mu_list)
    df.to_csv(outputfile, index=False, sep='\t')