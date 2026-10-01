import pandas as pd
import pytest
import torch
from pathlib import Path

from torch.utils.data import DataLoader

from modules_vae.model import MultiModalVAE, ShapMultiModalVAE
from modules_vae.param_grid import param_grid
from utils.dataset import Dataset

MUTATION_TRAIN_FILE = (
    Path(__file__).resolve().parents[1]
    / 'data/splits/0/0/train_features_os_processed_mut_nan.parquet'
)


def make_dataframe(mutation_features=23):
    data = {
        'survflag': [0, 1],
        'survtime': [1, 2],
        'Feature_exp_gene_a': [0.1, 0.2],
    }
    data.update({f'Feature_mut_gene_{i}': [0.0, 1.0] for i in range(mutation_features)})
    return pd.DataFrame(data)


def load_mutation_training_dataframe():
    dataframe = pd.read_parquet(MUTATION_TRAIN_FILE)
    return dataframe.assign(survflag=0, survtime=1)


def test_dataset_emits_mutation_matrix():
    dataset = Dataset(make_dataframe(), ['exp', 'mut'])

    batch = next(iter(DataLoader(dataset, batch_size=2)))

    assert dataset.X_mut.shape == (2, 23)
    assert batch['X_mut'].shape == (2, 23)
    assert batch['X_mut'].dtype == torch.float64


def test_dataset_selects_top_mutations_by_training_frequency():
    dataframe = pd.DataFrame({
        'survflag': [0, 1, 0],
        'survtime': [1, 2, 3],
        'Feature_mut_rare': [1.0, 0.0, 0.0],
        'Feature_mut_common': [1.0, 1.0, 0.0],
        'Feature_mut_middle': [1.0, 1.0, 1.0],
    })

    selected_dataframe, selected = Dataset.subset_mutation_features(
        dataframe, topKgenes=2
    )

    assert selected == ['Feature_mut_middle', 'Feature_mut_common']
    assert set(selected_dataframe.filter(regex='Feature_mut').columns) == set(selected)


def test_dataset_reuses_training_mutation_selection_for_inference():
    training = make_dataframe(mutation_features=3)
    inference = training.copy()
    selected = ['Feature_mut_gene_1']

    dataset = Dataset(
        inference,
        ['mut'],
        mutation_feature_columns=selected,
    )

    assert list(dataset.X_mut.shape) == [2, 1]


def test_dataset_without_mutation_input_preserves_existing_modalities():
    dataset = Dataset(make_dataframe(mutation_features=0), ['exp'])

    assert dataset.X_exp.shape == (2, 1)
    assert not hasattr(dataset, 'X_mut')


def test_dataset_rejects_zero_width_requested_input():
    with pytest.raises(ValueError, match=r"X_mut has width 0"):
        Dataset(make_dataframe(mutation_features=0), ['exp', 'mut'])


def test_real_mutation_training_parquet_emits_mutation_matrix():
    dataframe = load_mutation_training_dataframe()
    dataset = Dataset(dataframe, ['mut'])

    batch = next(iter(DataLoader(dataset, batch_size=4)))

    assert len(dataframe.filter(regex='Feature_mut').columns) == 23
    assert dataset.X_mut.shape == (len(dataframe), 23)
    assert batch['X_mut'].shape == (4, 23)


def test_real_mutation_training_parquet_rejects_missing_mutation_columns():
    dataframe = load_mutation_training_dataframe()
    mutation_columns = dataframe.filter(regex='Feature_mut').columns
    dataframe_without_mutation = dataframe.drop(columns=mutation_columns)

    with pytest.raises(ValueError, match=r"X_mut has width 0"):
        Dataset(dataframe_without_mutation, ['mut'])


def test_mutation_modality_is_supported_by_model_and_grid():
    model = MultiModalVAE(
        input_types=['exp', 'cna', 'gistic', 'fish', 'sbs', 'ig', 'mut'],
        input_dims=[720, 220, 80, 40, 20, 8, 23],
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
        input_dims=[720, 220, 80, 40, 20, 8, 23],
        layer_dims=[[1], [1], [1], [1], [1], [1], [4]],
        input_types_subtask=['clin'],
        input_dims_subtask=[4],
        layer_dims_subtask=[4,1],
    )
    shap_inputs = [
        torch.randn(2, input_dim, dtype=torch.float64)
        for input_dim in [720, 220, 80, 40, 20, 8, 23, 4]
    ]

    riskpred = model(shap_inputs)

    assert riskpred.shape == (2, 1)