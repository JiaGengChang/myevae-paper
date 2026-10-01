import torch
import pandas as pd
from pathlib import Path

from modules_vae.model import MultiModalVAE
from torch.utils.data import DataLoader
from utils.dataset import Dataset
from utils.type_prefixes import type_prefixes_dict


REAL_FEATURES_PATH = (
    Path(__file__).parents[1]
    / 'data/splits/0/0/train_features_pfs_processed_mut_nan.parquet'
)


def make_vae():
    return MultiModalVAE(
        input_types=['exp'],
        input_dims=[2],
        layer_dims=[[2]],
        input_types_subtask=['clin'],
        input_dims_subtask=[1],
        layer_dims_subtask=[2, 1],
        z_dim=3,
        activation=torch.nn.ReLU(),
        subtask_activation=torch.nn.Tanh(),
    )


def test_dataloader_emits_original_imputed_features_and_observed_mask():
    dataframe = pd.DataFrame({
        'survflag': [1, 0], 'survtime': [2.0, 3.0],
        'Feature_exp_a': [1.0, float('nan')], 'Feature_exp_b': [float('nan'), 4.0],
    })
    batch = next(iter(DataLoader(Dataset(dataframe, ['exp']), batch_size=2)))

    assert torch.isnan(batch['X_exp']).sum().item() == 2
    assert torch.equal(batch['X_exp_imputed'], torch.tensor([[1., 0.], [0., 4.]], dtype=torch.float64))
    assert torch.equal(batch['X_exp_mask'], torch.tensor([[1., 0.], [0., 1.]], dtype=torch.float64))


def test_model_receives_no_nan_from_zero_imputed_dataloader_features():
    dataframe = pd.DataFrame({
        'survflag': [1], 'survtime': [2.0],
        'Feature_exp_a': [float('nan')], 'Feature_exp_b': [4.0],
        'Feature_clin_a': [float('nan')],
    })
    batch = next(iter(DataLoader(Dataset(dataframe, ['exp', 'clin']), batch_size=1)))
    model = make_vae()
    captured = []
    model.encoders[0].register_forward_pre_hook(lambda _, args: captured.append(args[0]))
    model.eval()
    model(([batch['X_exp_imputed']], [batch['X_clin_imputed']]))

    assert not torch.isnan(captured[0]).any()


def test_reconstruction_loss_uses_only_observed_positions():
    output = torch.tensor([[100.0, 5.0], [7.0, 100.0]], dtype=torch.float64)
    target = torch.tensor([[1.0, float('nan')], [float('nan'), 4.0]], dtype=torch.float64)
    mask = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)

    loss = MultiModalVAE.reconstruction_loss(output, target, mask)
    expected = ((output[0, 0] - target[0, 0]) ** 2 + (output[1, 1] - target[1, 1]) ** 2) / 2

    assert torch.allclose(loss, expected)


def test_reconstruction_loss_uses_all_positions_without_mask():
    output = torch.tensor([[1.0, 5.0], [7.0, 4.0]], dtype=torch.float64)
    target = torch.tensor([[1.0, 1.0], [5.0, 4.0]], dtype=torch.float64)

    loss = MultiModalVAE.reconstruction_loss(output, target, None)
    expected = ((output - target) ** 2).mean()

    assert torch.allclose(loss, expected)


def test_real_features_emit_masks_and_masked_reconstruction_loss():
    dataframe = pd.read_parquet(REAL_FEATURES_PATH).copy()
    dataframe['survflag'] = 1
    dataframe['survtime'] = 2.0
    sbs_columns = dataframe.filter(regex=type_prefixes_dict['sbs']).columns

    dataset = Dataset(dataframe, ['sbs'])
    batch = next(iter(DataLoader(dataset, batch_size=len(dataset))))
    expected_mask = torch.tensor(
        (~dataframe.loc[:, sbs_columns].isna()).to_numpy(),
        dtype=torch.float64,
    )

    assert torch.equal(batch['X_sbs_mask'], expected_mask)
    assert torch.equal(
        batch['X_sbs_imputed'],
        torch.tensor(
            dataframe.loc[:, sbs_columns].fillna(0.0).to_numpy(),
            dtype=torch.float64,
        ),
    )

    output = torch.where(
        expected_mask.bool(),
        batch['X_sbs'],
        torch.full_like(batch['X_sbs'], 1000.0),
    )
    loss = MultiModalVAE.reconstruction_loss(
        output,
        batch['X_sbs'],
        batch['X_sbs_mask'],
    )
    print(f'reconstruction loss: {loss.item()}')

    assert torch.allclose(loss, torch.tensor(0.0, dtype=torch.float64))
