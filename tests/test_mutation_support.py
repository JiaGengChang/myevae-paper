import pandas as pd
import torch

from torch.utils.data import DataLoader

from modules_vae.model import MultiModalVAE, ShapMultiModalVAE
from modules_vae.param_grid import param_grid
from utils.dataset import Dataset


def make_dataframe(mutation_features=23):
    data = {
        'survflag': [0, 1],
        'survtime': [1, 2],
        'Feature_exp_gene_a': [0.1, 0.2],
    }
    data.update({f'Feature_mut_gene_{i}': [0.0, 1.0] for i in range(mutation_features)})
    return pd.DataFrame(data)


def test_dataset_emits_mutation_matrix():
    dataset = Dataset(make_dataframe(), ['exp', 'mut'])

    batch = next(iter(DataLoader(dataset, batch_size=2)))

    assert dataset.X_mut.shape == (2, 23)
    assert batch['X_mut'].shape == (2, 23)
    assert batch['X_mut'].dtype == torch.float64


def test_dataset_without_mutation_input_preserves_existing_modalities():
    dataset = Dataset(make_dataframe(mutation_features=0), ['exp'])

    assert dataset.X_exp.shape == (2, 1)
    assert not hasattr(dataset, 'X_mut')


def test_mutation_modality_is_supported_by_model_and_grid():
    model = MultiModalVAE(
        input_types=['exp', 'cna', 'gistic', 'fish', 'sbs', 'ig', 'mut'],
        input_dims=[1, 1, 1, 1, 1, 1, 23],
        layer_dims=[[1], [1], [1], [1], [1], [1], [4]],
        input_types_subtask=['clin'],
        input_dims_subtask=[0],
        layer_dims_subtask=[1],
    )

    assert model.input_types_vae == ['exp', 'cna', 'gistic', 'fish', 'sbs', 'ig', 'mut']
    assert param_grid['input_types'][0][-1] == 'mut'
    assert param_grid['layer_dims'][0][-1] == [4]


def test_shap_model_accepts_all_vae_modalities_including_mutation():
    model = ShapMultiModalVAE(
        input_types=['exp', 'cna', 'gistic', 'fish', 'sbs', 'ig', 'mut'],
        input_dims=[1, 1, 1, 1, 1, 1, 23],
        layer_dims=[[1], [1], [1], [1], [1], [1], [4]],
        input_types_subtask=['clin'],
        input_dims_subtask=[0],
        layer_dims_subtask=[1],
    )
    shap_inputs = [
        torch.randn(2, input_dim, dtype=torch.float64)
        for input_dim in [1, 1, 1, 1, 1, 1, 23, 0]
    ]

    riskpred = model(shap_inputs)

    assert riskpred.shape == (2, 1)