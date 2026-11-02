"""The pixel-wise L1 objective used for joint T1/T2 supervision."""
import torch
from torch import nn


class TSFDLoss(nn.Module):
    def forward(self, prediction, target, mask=None):
        if prediction.shape != target.shape:
            raise ValueError('Prediction and target shapes must match')
        error = torch.abs(prediction - target)
        if mask is not None:
            if mask.shape != (target.shape[0], 1, *target.shape[2:]):
                raise ValueError('Mask must have shape [B, 1, H, W]')
            error = error * mask
        # Match the original full-tensor mean, also when a mask is supplied.
        return error.mean()
