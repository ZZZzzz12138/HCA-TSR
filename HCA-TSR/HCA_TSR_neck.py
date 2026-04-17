# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.model import BaseModule
from mmdet.registry import MODELS


# -----------------------------
# Utility function: optional normalization
# -----------------------------
def build_norm(num_channels: int, norm_type: str = 'BN', **kwargs):
    norm_type = (norm_type or 'bn').lower()
    if norm_type == 'bn':
        return nn.BatchNorm2d(num_channels)
    if norm_type == 'syncbn':
        return nn.SyncBatchNorm(num_channels)
    if norm_type == 'gn':
        # Adaptively choose a valid number of groups for GroupNorm
        default_groups = kwargs.get('groups', 32)
        groups = min(default_groups, num_channels)
        groups = math.gcd(groups, num_channels)
        if groups == 0:
            groups = 1
        while num_channels % groups != 0 and groups > 1:
            groups //= 2
        return nn.GroupNorm(groups, num_channels)
    if norm_type in ('none', 'identity', None):
        return nn.Identity()
    raise ValueError(f'Unknown norm_type: {norm_type}')


# -----------------------------
# Lightweight block: depthwise separable convolution
# -----------------------------
class DWSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None, norm='BN', act=True):
        super().__init__()
        if p is None:
            p = k // 2
        self.dw = nn.Conv2d(
            in_ch, in_ch, kernel_size=k, stride=s, padding=p,
            groups=in_ch, bias=False
        )
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = build_norm(out_ch, norm)
        self.act = nn.ReLU(inplace=False) if act else nn.Identity()

        nn.init.kaiming_normal_(self.dw.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.pw.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        x = self.act(x)
        return x


# -----------------------------
# Lightweight spatial gate: DWConv 5x5 + Sigmoid
# -----------------------------
class SpatialGateLite(nn.Module):
    def __init__(self, channels, norm='BN'):
        super().__init__()
        # Keep the original design to preserve behavior
        self.reduce = nn.AdaptiveAvgPool2d((None, None))
        self.dw = nn.Conv2d(
            channels, channels, kernel_size=5, padding=2,
            groups=channels, bias=False
        )
        self.bn = build_norm(channels, norm)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

        nn.init.kaiming_normal_(self.dw.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.proj.bias, 0.)

    def forward(self, x):
        y = self.reduce(x)
        y = self.dw(y)
        y = self.bn(y)
        y = self.proj(y)
        return torch.sigmoid(y)


# -----------------------------
# Detail extractor: learnable Laplacian + depthwise separable convolution
# -----------------------------
class DetailExtractor(nn.Module):
    def __init__(self, channels, norm='BN'):
        super().__init__()
        self.kernel = nn.Parameter(
            torch.tensor([[0., -1., 0.],
                          [-1., 4., -1.],
                          [0., -1., 0.]], dtype=torch.float32).view(1, 1, 3, 3),
            requires_grad=True
        )
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn = build_norm(channels, norm)
        self.act = nn.ReLU(inplace=False)

        nn.init.kaiming_normal_(self.dw.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.pw.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        k = self.kernel.expand(x.size(1), 1, 3, 3)
        y = F.conv2d(x, k, padding=1, groups=x.size(1))
        y = self.dw(y)
        y = self.pw(y)
        y = self.bn(y)
        y = self.act(y)
        return y


# -----------------------------
# HCA-TSR core fusion block
# Note:
# Python class names cannot contain hyphens, so the identifier is HCATSR.
# The registry name exposed to MMDetection is set to 'HCA-TSR'.
# -----------------------------
class HCATSR(nn.Module):
    """HCA-TSR core module.

    Inputs:
        x1: high-resolution feature, shape (N, C1, H, W)
        x2: low-resolution feature, shape (N, C2, h, w)
    """

    def __init__(
        self,
        input_channel1: int,
        input_channel2: int,
        norm_type: str = 'BN',
        gamma: int = 2,
        bias: int = 1,
        gate_eps: float = 1e-6,
        init_tau: float = 1.5,
        debug: bool = False,
    ):
        super().__init__()
        self.c1 = input_channel1
        self.c2 = input_channel2
        self.debug = debug
        self.gate_eps = gate_eps

        # ECA-style adaptive kernel size
        def calc_k(c):
            k = int((math.log(c, 2) + bias) / gamma)
            if k < 3:
                k = 3
            return k if k % 2 == 1 else k + 1

        k1 = calc_k(self.c1)
        k2 = calc_k(self.c2)
        k3 = calc_k(self.c1 + self.c2)

        self.avg1 = nn.AdaptiveAvgPool2d(1)
        self.avg2 = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv1d(1, 1, k1, padding=k1 // 2, bias=False)
        self.conv2 = nn.Conv1d(1, 1, k2, padding=k2 // 2, bias=False)
        self.conv3 = nn.Conv1d(1, 1, k3, padding=k3 // 2, bias=False)

        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv3.weight, mode='fan_out', nonlinearity='relu')

        # Learnable temperature parameter constrained to [0.5, 2.0]
        self._tau_raw = nn.Parameter(torch.tensor(init_tau, dtype=torch.float32))
        self._tau_min, self._tau_max = 0.5, 2.0

        # Branch projections
        self.proj_x1 = DWSeparableConv(self.c1, self.c1, k=3, norm=norm_type, act=True)
        self.proj_x2 = DWSeparableConv(self.c2, self.c1, k=3, norm=norm_type, act=True)

        # Spatial gating for the upsampled branch
        self.spatial_gate = SpatialGateLite(self.c1, norm=norm_type)

        # High-frequency detail branch from x1
        self.detail = DetailExtractor(self.c1, norm=norm_type)
        self.detail_scale = nn.Parameter(torch.tensor(0.3, dtype=torch.float32))

        # Learnable three-branch fusion weights
        self.w_x1 = nn.Parameter(torch.tensor(1.0))
        self.w_x2 = nn.Parameter(torch.tensor(1.0))
        self.w_dt = nn.Parameter(torch.tensor(0.5))
        self._eps = 1e-4

    def _finite(self, x):
        return torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)

    def _tau(self):
        return torch.clamp(self._tau_raw, self._tau_min, self._tau_max)

    def forward(self, x1, x2):
        n = x1.size(0)
        x1 = self._finite(x1)
        x2 = self._finite(x2)

        # Joint channel gating
        g1 = self.avg1(x1).view(n, 1, self.c1)
        g2 = self.avg2(x2).view(n, 1, self.c2)
        g1 = self.conv1(g1).view(n, self.c1, 1, 1)
        g2 = self.conv2(g2).view(n, self.c2, 1, 1)

        g_cat = torch.cat([g1, g2], dim=1).view(n, 1, self.c1 + self.c2)
        g_cat = self.conv3(g_cat).view(n, self.c1 + self.c2, 1, 1)

        tau = self._tau()
        gate = torch.sigmoid(g_cat / tau)
        gate = self._finite(gate).clamp(self.gate_eps, 1.0 - self.gate_eps)
        g1, g2 = torch.split(gate, [self.c1, self.c2], dim=1)

        # Branch preprocessing
        x1_p = self.proj_x1(self._finite(x1 * g1))
        target_hw = x1.shape[-2:]
        x2_up = F.interpolate(
            self._finite(x2 * g2),
            size=target_hw,
            mode='bilinear',
            align_corners=False
        )
        x2_p = self.proj_x2(x2_up)

        # Spatial gating on the upsampled branch
        sgate = self.spatial_gate(x2_p)
        x2_p = x2_p * sgate

        # Detail branch
        detail = self.detail(x1) * torch.sigmoid(self.detail_scale)

        # Three-branch weighted fusion
        w = torch.stack([
            F.softplus(self.w_x1),
            F.softplus(self.w_x2),
            F.softplus(self.w_dt)
        ], dim=0) + self._eps
        w = w / w.sum()

        out = w[0] * x1_p + w[1] * x2_p + w[2] * detail
        out = self._finite(out)
        return out


# -----------------------------
# HCA-TSR neck
# The registry name is 'HCA-TSR' so it can be referenced directly in configs.
# -----------------------------
@MODELS.register_module(name='HCA-TSR')
class HCATSRNeck(BaseModule):
    """HCA-TSR pyramid neck with an FPN-compatible interface.

    Args:
        in_channels (List[int]): Backbone output channels, e.g. [256, 512, 1024, 2048].
        out_channels (int): Unified output channel size.
        norm_type (str): 'BN' | 'SyncBN' | 'GN' | 'None'
        debug (bool): Whether to print debug information.

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]:
            (P3, P4, P5, P6) with strides approximately [8, 16, 32, 64].
    """

    def __init__(
        self,
        in_channels,
        out_channels: int = 256,
        norm_type: str = 'BN',
        init_cfg=None,
        debug: bool = False,
    ):
        super().__init__(init_cfg=init_cfg)
        assert len(in_channels) >= 2, 'HCA-TSR requires at least two input feature levels.'

        self.num_levels = len(in_channels)
        self.out_channels = out_channels
        self.debug = debug

        # Native top-down fusion blocks
        self.blocks = nn.ModuleList([
            HCATSR(in_channels[i], in_channels[i + 1], norm_type=norm_type)
            for i in range(self.num_levels - 1)
        ])

        # Project fused features to the same output dimension
        self.out_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(cin, out_channels, 3, padding=1, bias=False),
                build_norm(out_channels, norm_type),
                nn.ReLU(inplace=False),
            )
            for cin in in_channels
        ])

        # Downsample P5 to produce P6
        self.p6_down = nn.Conv2d(
            out_channels, out_channels, 3, stride=2, padding=1, bias=False
        )
        nn.init.kaiming_normal_(self.p6_down.weight, mode='fan_out', nonlinearity='relu')

        # Initialize convolution layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if getattr(m, 'bias', None) is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        if self.debug:
            print('Debug: Checking input format')
            if isinstance(inputs, (list, tuple)):
                print(f'Inputs are of type: {type(inputs)} and length: {len(inputs)}')
                for i, input_tensor in enumerate(inputs):
                    print(f'Input {i} shape: {input_tensor.shape}')
            else:
                print(f'Inputs are of type: {type(inputs)}. Expected list or tuple.')

        assert isinstance(inputs, (list, tuple)) and len(inputs) == self.num_levels

        # Fuse from top to bottom: C5 -> C4' -> C3' -> C2'
        native = [inputs[-1]]
        x = inputs[-1]
        for i in reversed(range(self.num_levels - 1)):
            x = self.blocks[i](inputs[i], x)
            native.append(x)
        native = list(reversed(native))

        outs = [self.out_convs[i](feat) for i, feat in enumerate(native)]
        assert len(outs) >= 4

        p3, p4, p5 = outs[1], outs[2], outs[3]
        p6 = self.p6_down(p5)
        return (p3, p4, p5, p6)
