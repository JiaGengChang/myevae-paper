import pandas as pd
import torch

from modules_vae.model import MultiModalVAE
from utils.dataset import Dataset
from utils.missing import masked_mse


def make_dataset():
    frame = pd.DataFrame({
        'Feature_exp_a': [0.0, float('nan'), 2.0],
        'Feature_exp_b': [1.0, float('inf'), float('nan')],
        'Feature_clin_a': [0.0, 1.0, 2.0],
        'survflag': [1, 0, 1],
        'survtime': [3.0, 4.0, 5.0],
    })
    return Dataset(frame, ['exp', 'clin'])


def make_model():
    return MultiModalVAE(
        input_types=['exp'], input_dims=[2], layer_dims=[[3]],
        input_types_subtask=['clin'], input_dims_subtask=[1],
        layer_dims_subtask=[2, 1], z_dim=2,
    )


def test_dataset_distinguishes_observed_zero_from_missing_zero():
    dataset = make_dataset()

    assert torch.equal(dataset.X_exp[0], torch.tensor([0.0, 1.0], dtype=torch.float64))
    assert torch.equal(dataset.X_exp[1], torch.tensor([0.0, 0.0], dtype=torch.float64))
    assert dataset.X_mask_exp[0, 0] == 1
    assert dataset.X_mask_exp[1, 0] == 0


def test_mask_aware_forward_and_gradient_are_finite():
    dataset = make_dataset()
    model = make_model()
    outputs, mu, logvar, risk = model((
        [(dataset.X_exp, dataset.X_mask_exp)],
        [(dataset.X_clin, dataset.X_mask_clin)],
    ))

    assert all(torch.isfinite(value).all() for value in [*outputs, mu, logvar, risk])
    loss = masked_mse(outputs[0], dataset.X_exp, dataset.X_mask_exp) + risk.square().mean()
    loss.backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_masked_mse_is_zero_for_all_missing_targets():
    value = masked_mse(
        torch.ones(1, 2, dtype=torch.float64),
        torch.zeros(1, 2, dtype=torch.float64),
        torch.zeros(1, 2, dtype=torch.float64),
    )
    assert value.item() == 0.0


def test_first_layer_matches_data_and_mask_affine():
    model = make_model()
    layer = model.encoder_exp.network[0]
    with torch.no_grad():
        layer.weight.copy_(torch.arange(layer.weight.numel(), dtype=torch.float64).reshape_as(layer.weight))
        layer.bias.zero_()
    x_zero = torch.tensor([[0.0, 2.0]], dtype=torch.float64)
    mask = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
    expected = layer.weight[:, :2] @ x_zero.T + layer.weight[:, 2:] @ mask.T
    actual = layer(torch.cat((x_zero, mask), dim=1)).T
    assert torch.allclose(actual, expected)
