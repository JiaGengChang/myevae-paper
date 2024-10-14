import torch
import numpy as np
import pandas as pd

def predict_to_tsv(model, loader, outputfile):
    model.eval()
    predictions = []
    public_ids = []
    for batch_idx, data in enumerate(loader):
        with torch.no_grad():
            inputs_vae = [data[f'X_{input_type}'] for input_type in model.input_types_vae]
            inputs_task = [data[f'X_{input_type}'] for input_type in model.input_types_subtask]
            _, _, _, riskpred = model.forward((inputs_vae, inputs_task))
            predictions.append(riskpred.detach().numpy())
            public_ids.append(data['PUBLIC_ID'])
    predictions = np.concatenate(predictions)
    public_ids = np.concatenate(public_ids)
    df = pd.DataFrame({
        'PUBLIC_ID': public_ids,
        'prediction': predictions.flatten()
    })
    df.to_csv(outputfile, index=False, sep='\t')