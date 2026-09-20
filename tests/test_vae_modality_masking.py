import pandas as pd
import torch

from modules_vae.model import MultiModalVAE
from utils.dataset import Dataset


def make_model(seed=None):
    return MultiModalVAE(
        input_types=['exp', 'cna'],
        input_dims=[3, 2],
        layer_dims=[[4], [3]],
        input_types_subtask=['clin'],
        input_dims_subtask=[1],
        layer_dims_subtask=[2, 1],
        z_dim=2,
        modality_mask_seed=seed,
    )


def test_target_modality_is_excluded_and_only_target_is_decoded():
    model = make_model()
    model.eval()
    inputs = [torch.ones(4, 3, dtype=torch.float64), torch.ones(4, 2, dtype=torch.float64)]
    task = [torch.ones(4, 1, dtype=torch.float64)]

    outputs, mu, _, _ = model((inputs, task), target_modality='exp')
    changed_target = [inputs[0] + 100, inputs[1]]
    _, changed_mu, _, _ = model((changed_target, task), target_modality='exp')

    assert len(outputs) == 1
    assert outputs[0].shape == inputs[0].shape
    assert torch.equal(mu, changed_mu)


def test_target_sampling_is_reproducible():
    first = make_model(seed=11)
    second = make_model(seed=11)

    assert [first.sample_target_modality() for _ in range(10)] == [
        second.sample_target_modality() for _ in range(10)
    ]


def test_dataset_zero_imputes_features_without_imputing_labels():
    frame = pd.DataFrame({
        'survflag': [1, 0],
        'survtime': [3.0, 5.0],
        'Feature_exp_a': [0.0, float('nan')],
        'Feature_CNA_ENSG_a': [float('nan'), 2.0],
    })
    dataset = Dataset(frame, ['exp', 'cna'])

    assert torch.equal(dataset.X_exp, torch.tensor([[0.0], [0.0]], dtype=torch.float64))
    assert torch.equal(dataset.X_cna, torch.tensor([[0.0], [2.0]], dtype=torch.float64))
    assert list(dataset.event_indicator) == [1, 0]
    assert list(dataset.event_time) == [3.0, 5.0]


def test_masked_forward_has_finite_gradients():
    model = make_model(seed=3)
    inputs = [torch.randn(5, 3, dtype=torch.float64), torch.randn(5, 2, dtype=torch.float64)]
    task = [torch.randn(5, 1, dtype=torch.float64)]

    outputs, mu, logvar, risk = model((inputs, task), target_modality='cna')
    loss = outputs[0].square().mean() + mu.square().mean() + logvar.square().mean() + risk.square().mean()
    loss.backward()

    assert torch.isfinite(loss)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
