from pathlib import Path

import pandas as pd
import torch
import pytest

from modules_vae.model import MultiModalVAE
from utils.dataset import Dataset


ACTUAL_FEATURES = Path(__file__).parents[1] / 'data/splits/0/0/valid_features_os_processed_mut_nan.parquet'
ACTUAL_MODALITIES = ['exp', 'cna', 'fish', 'sbs', 'gistic', 'ig']


def make_vae(masking_proportions=None):
    return MultiModalVAE(
        input_types=['exp', 'cna', 'fish', 'sbs', 'gistic', 'ig'],
        input_dims=[4, 3, 2, 2, 2, 2],
        layer_dims=[[2], [2], [2], [2], [2], [2]],
        input_types_subtask=['clin'],
        input_dims_subtask=[1],
        layer_dims_subtask=[2, 1],
        z_dim=3,
        activation=torch.nn.ReLU(),
        subtask_activation=torch.nn.Tanh(),
        masking_proportions=masking_proportions or {
            'exp': 0.25,
            'cna': 0.5,
            'fish': 0.5,
            'sbs': 0.5,
            'gistic': 0.5,
            'ig': 0.5,
        },
        random_state=7,
    )


def make_actual_dataset():
    dataframe = pd.read_parquet(ACTUAL_FEATURES).iloc[:8].copy()
    dataframe['survflag'] = 0
    dataframe['survtime'] = 1.0
    return Dataset(dataframe, ACTUAL_MODALITIES + ['clin'])


def make_actual_vae(dataset):
    input_dims = [getattr(dataset, f'X_{modality}').shape[1] for modality in ACTUAL_MODALITIES]
    return MultiModalVAE(
        input_types=ACTUAL_MODALITIES,
        input_dims=input_dims,
        layer_dims=[[4] for _ in ACTUAL_MODALITIES],
        input_types_subtask=['clin'],
        input_dims_subtask=[dataset.X_clin.shape[1]],
        layer_dims_subtask=[2, 1],
        z_dim=3,
        activation=torch.nn.ReLU(),
        subtask_activation=torch.nn.Tanh(),
        masking_proportions={modality: 0.5 for modality in ACTUAL_MODALITIES},
        random_state=7,
    )


def test_mask_generation_is_seeded_and_bernoulli_scaled_per_modality():
    vae_1 = make_vae()
    vae_2 = make_vae()
    x = torch.ones((50, 20), dtype=torch.float64)

    mask_1 = vae_1._sample_modality_mask(x, 0.25, generator=vae_1.mask_generator)
    mask_2 = vae_2._sample_modality_mask(x, 0.25, generator=vae_2.mask_generator)

    assert mask_1.shape == x.shape
    assert mask_1.dtype == torch.float64
    assert torch.all(mask_1 >= 0)
    assert torch.all(mask_1 <= 1)
    assert torch.all((mask_1 == 0) | (mask_1 == 1))
    assert abs(mask_1.sum().item() / x.numel() - 0.25) < 0.15
    assert torch.equal(mask_1, mask_2)


def test_masking_config_validation_rejects_unknown_or_missing_modalities():
    vae = make_vae()

    with pytest.raises(ValueError, match='Unknown modality'):
        vae.validate_masking_config({'exp': 0.1, 'unknown': 0.2}, ['exp', 'cna'])

    with pytest.raises(ValueError, match='missing'):
        vae.validate_masking_config({'exp': 0.1}, ['exp', 'cna'])


def test_masked_batch_keeps_targets_and_zeroes_masked_features():
    vae = make_vae()
    x_exp = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=torch.float64)
    x_cna = torch.tensor([[0.2, 0.4, 0.6], [1.2, 1.4, 1.6]], dtype=torch.float64)
    x_fish = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float64)
    x_sbs = torch.tensor([[0.5, 0.6], [0.7, 0.8]], dtype=torch.float64)
    x_gistic = torch.tensor([[0.9, 1.0], [1.1, 1.2]], dtype=torch.float64)
    x_ig = torch.tensor([[1.3, 1.4], [1.5, 1.6]], dtype=torch.float64)
    inputs = [x_exp, x_cna, x_fish, x_sbs, x_gistic, x_ig]
    proportions = {
        'exp': 0.25,
        'cna': 0.5,
        'fish': 0.5,
        'sbs': 0.5,
        'gistic': 0.5,
        'ig': 0.5,
    }

    masked, targets, masks, stats = vae.apply_mask_to_batch(inputs, proportions)

    assert targets[0].shape == x_exp.shape
    assert torch.equal(targets[0], x_exp)
    assert len(masked) == len(inputs)
    assert len(targets) == len(inputs)
    assert len(masks) == len(inputs)
    assert all(mask.shape == input_tensor.shape for mask, input_tensor in zip(masks, inputs))
    assert all(
        torch.all(masked_input == input_tensor * (1.0 - mask))
        for masked_input, input_tensor, mask in zip(masked, inputs, masks)
    )
    assert stats['exp']['mask_rate'] >= 0.0
    assert stats['cna']['mask_rate'] >= 0.0
    assert stats['fish']['mask_rate'] >= 0.0
    assert stats['sbs']['mask_rate'] >= 0.0
    assert stats['gistic']['mask_rate'] >= 0.0
    assert stats['ig']['mask_rate'] >= 0.0


def test_masked_reconstruction_loss_uses_only_selected_features():
    output = torch.tensor([[1.0, 5.0, 2.0], [1.0, 0.0, 3.0]], dtype=torch.float64)
    target = torch.tensor([[0.0, 5.0, 5.0], [1.0, 1.0, 1.0]], dtype=torch.float64)
    mask = torch.tensor([[0.0, 1.0, 1.0], [0.0, 0.0, 1.0]], dtype=torch.float64)

    loss = MultiModalVAE.reconstruction_loss(output, target, mask)
    expected = ((output - target) ** 2 * mask).sum() / mask.sum().clamp_min(1.0)

    assert torch.allclose(loss, expected)


def test_actual_dataframe_inputs_are_masked_per_modality():
    dataset = make_actual_dataset()
    vae = make_actual_vae(dataset)
    inputs = [getattr(dataset, f'X_{modality}') for modality in ACTUAL_MODALITIES]

    masked, targets, masks, stats = vae.apply_mask_to_batch(inputs)

    assert [tensor.shape[1] for tensor in targets] == [713, 216, 36, 10, 64, 8]
    for input_tensor, masked_tensor, target, mask, modality in zip(
        inputs, masked, targets, masks, ACTUAL_MODALITIES
    ):
        expected_target = torch.nan_to_num(input_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        assert torch.equal(target, expected_target)
        assert torch.equal(masked_tensor, target * (1.0 - mask))
        assert torch.all((mask == 0) | (mask == 1))
        assert stats[modality]['count'] == int(mask.sum().item())
        assert stats[modality]['total_features'] == mask.numel()
        assert stats[modality]['count'] > 0


def test_actual_dataframe_reconstruction_loss_ignores_unobserved_positions():
    dataset = make_actual_dataset()
    vae = make_actual_vae(dataset)
    inputs = [getattr(dataset, f'X_{modality}') for modality in ACTUAL_MODALITIES]
    _, targets, masks, _ = vae.apply_mask_to_batch(inputs)

    target = targets[0]
    mask = masks[0]
    output = target.clone()
    output[mask == 1] += 2.0
    loss = vae.reconstruction_loss(output, target, mask)

    changed_outside_mask = output.clone()
    changed_outside_mask[mask == 0] += 1000.0
    assert torch.allclose(
        loss,
        vae.reconstruction_loss(changed_outside_mask, target, mask),
    )
    assert loss > 0
