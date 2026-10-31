import torch
import torch.nn as nn

from . import DWT_tfmamba_MPU_v5 as base


class ZeroFilterLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return torch.zeros_like(x)


class NoWaveletBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, H, W):
        return x


class IdentitySwinLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


class IdentityTemporalLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return torch.zeros_like(x)


class NoCIFMProjectionUnit(base.ManifoldProjectionUnit):
    def forward(self, x_mamba, x_swin):
        return self.proj_m(x_mamba), self.proj_s(x_swin)


def _instantiate_with_patch(**patches):
    original = {name: getattr(base, name) for name in patches}
    try:
        for name, value in patches.items():
            setattr(base, name, value)
        return base.CSBNet()
    finally:
        for name, value in original.items():
            setattr(base, name, value)


def CSBNet_no_lwe():
    return _instantiate_with_patch(TrueWaveletBlock=NoWaveletBlock)


def CSBNet_no_gsf():
    return _instantiate_with_patch(GlobalFilterLayer=ZeroFilterLayer)


def CSBNet_no_cifm():
    return _instantiate_with_patch(ManifoldProjectionUnit=NoCIFMProjectionUnit)


def CSBNet_no_swin():
    return _instantiate_with_patch(SwinLayer=IdentitySwinLayer)


def CSBNet_no_ptmamba():
    return _instantiate_with_patch(BiTemporalMambaLayer=IdentityTemporalLayer)
