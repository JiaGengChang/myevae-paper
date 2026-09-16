import torch


def masked_mse(prediction, target_zero, mask):
    """Compute MSE over observed targets; return zero for an all-missing target."""
    squared_error = (prediction - target_zero).pow(2) * mask
    return squared_error.sum() / mask.sum().clamp_min(1.0)


def input_pair(data, input_type):
    return data[f'X_{input_type}'], data[f'X_mask_{input_type}']


def tensor_pair(values):
    mask = torch.isfinite(values).to(dtype=values.dtype)
    return torch.where(mask.bool(), values, torch.zeros_like(values)), mask
