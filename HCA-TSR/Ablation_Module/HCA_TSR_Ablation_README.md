# English | [简体中文](#简体中文)

This repository contains **six ablation neck modules** for the **HCA-TSR** design under the **MMDetection** framework.  
These files are designed for controlled ablation experiments in object detection pipelines, especially for evaluating the contribution of each sub-module in the HCA-TSR neck.

---

## English

### 1. Overview

The following six ablation modules are included:

| Ablation Target | Registry Name | Python Neck Class | File Name | Description |
|---|---|---|---|---|
| Remove ChannelGate | `HCA-TSR-NoChannelGate` | `HCATSRNoChannelGateNeck` | `HCA_TSR_NoChannelGate.py` | Removes channel gating in the HCA branch |
| Remove DetailBypass | `HCA-TSR-NoDetailBypass` | `HCATSRNoDetailBypassNeck` | `HCA_TSR_NoDetailBypass.py` | Removes the detail enhancement bypass branch |
| Remove SpatialGate | `HCA-TSR-NoSpatialGate` | `HCATSRNoSpatialGateNeck` | `HCA_TSR_NoSpatialGate.py` | Removes spatial gating in the TSR branch |
| Remove Tau | `HCA-TSR-NoTau` | `HCATSRNoTauNeck` | `HCA_TSR_NoTau.py` | Removes the temperature coefficient `tau` in channel gating |
| Fixed Blend Weights | `HCA-TSR-FixedBlendWeights` | `HCATSRFixedBlendWeightsNeck` | `HCA_TSR_FixedBlendWeights.py` | Replaces learnable fusion weights with fixed constants |
| Replace DW with StdConv | `HCA-TSR-StdConv` | `HCATSRStdConvNeck` | `HCA_TSR_StdConv.py` | Replaces depthwise separable convolutions with standard convolutions |

These modules are designed to be used as **drop-in neck replacements** inside MMDetection config files.

---

### 2. Recommended Directory Structure in MMDetection

Place the ablation module files under your custom neck directory, for example:

```text
mmdetection/
├─ mmdet/
│  ├─ models/
│  │  ├─ necks/
│  │  │  ├─ HCA_TSR_NoChannelGate.py
│  │  │  ├─ HCA_TSR_NoDetailBypass.py
│  │  │  ├─ HCA_TSR_NoSpatialGate.py
│  │  │  ├─ HCA_TSR_NoTau.py
│  │  │  ├─ HCA_TSR_FixedBlendWeights.py
│  │  │  ├─ HCA_TSR_StdConv.py
│  │  │  └─ __init__.py
```

Then import them in `mmdet/models/necks/__init__.py`:

```python
from .HCA_TSR_NoChannelGate import HCATSRNoChannelGateNeck
from .HCA_TSR_NoDetailBypass import HCATSRNoDetailBypassNeck
from .HCA_TSR_NoSpatialGate import HCATSRNoSpatialGateNeck
from .HCA_TSR_NoTau import HCATSRNoTauNeck
from .HCA_TSR_FixedBlendWeights import HCATSRFixedBlendWeightsNeck
from .HCA_TSR_StdConv import HCATSRStdConvNeck

__all__ = [
    'HCATSRNoChannelGateNeck',
    'HCATSRNoDetailBypassNeck',
    'HCATSRNoSpatialGateNeck',
    'HCATSRNoTauNeck',
    'HCATSRFixedBlendWeightsNeck',
    'HCATSRStdConvNeck',
]
```

---

### 3. How MMDetection Uses These Modules

Each file registers a neck through:

```python
@MODELS.register_module(name='...')
```

This means the config file should reference the **registry name**, not the Python class name.

Example:

```python
neck=dict(
    type='HCA-TSR-NoChannelGate',
    in_channels=[256, 512, 1024, 2048],
    out_channels=256,
    norm_type='BN'
)
```

---

### 4. Configuration Examples

#### 4.1 NoChannelGate

```python
model = dict(
    neck=dict(
        type='HCA-TSR-NoChannelGate',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        norm_type='BN'
    )
)
```

#### 4.2 NoDetailBypass

```python
model = dict(
    neck=dict(
        type='HCA-TSR-NoDetailBypass',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        norm_type='BN'
    )
)
```

#### 4.3 NoSpatialGate

```python
model = dict(
    neck=dict(
        type='HCA-TSR-NoSpatialGate',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        norm_type='BN'
    )
)
```

#### 4.4 NoTau

```python
model = dict(
    neck=dict(
        type='HCA-TSR-NoTau',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        norm_type='BN'
    )
)
```

#### 4.5 FixedBlendWeights

```python
model = dict(
    neck=dict(
        type='HCA-TSR-FixedBlendWeights',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        norm_type='BN',
        fixed_weights=(0.6, 0.3, 0.1)
    )
)
```

#### 4.6 StdConv

```python
model = dict(
    neck=dict(
        type='HCA-TSR-StdConv',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        norm_type='BN'
    )
)
```

---

### 5. Typical Backbone Compatibility

These ablation necks follow an **FPN-like multi-level interface** and are suitable for common MMDetection backbones that output four feature stages, such as:

- ResNet-50 / ResNet-101
- ResNeXt
- EfficientNet backbones adapted to MMDetection
- Other backbones that provide four pyramid inputs

Typical channel setting:

```python
in_channels=[256, 512, 1024, 2048]
out_channels=256
```

If your backbone outputs different channel dimensions, modify `in_channels` accordingly.

---

### 6. Output Feature Levels

All six ablation necks return:

```python
(P3, P4, P5, P6)
```

Approximate strides:

- `P3`: stride 8
- `P4`: stride 16
- `P5`: stride 32
- `P6`: stride 64

This makes them directly usable in many one-stage and two-stage detectors in MMDetection.

---

### 7. Example: Using with RetinaNet

```python
model = dict(
    type='RetinaNet',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
    ),
    neck=dict(
        type='HCA-TSR-NoSpatialGate',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        norm_type='BN'
    ),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=26,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64]
        ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]
        ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0
        ),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)
    )
)
```

---

### 8. Example: Switching Between Ablation Variants

Ablation experiments are easiest if only the `type` field changes.

```python
# NoChannelGate
neck=dict(type='HCA-TSR-NoChannelGate', in_channels=[256, 512, 1024, 2048], out_channels=256, norm_type='BN')

# NoDetailBypass
neck=dict(type='HCA-TSR-NoDetailBypass', in_channels=[256, 512, 1024, 2048], out_channels=256, norm_type='BN')

# NoSpatialGate
neck=dict(type='HCA-TSR-NoSpatialGate', in_channels=[256, 512, 1024, 2048], out_channels=256, norm_type='BN')

# NoTau
neck=dict(type='HCA-TSR-NoTau', in_channels=[256, 512, 1024, 2048], out_channels=256, norm_type='BN')

# FixedBlendWeights
neck=dict(type='HCA-TSR-FixedBlendWeights', in_channels=[256, 512, 1024, 2048], out_channels=256, norm_type='BN', fixed_weights=(0.6, 0.3, 0.1))

# StdConv
neck=dict(type='HCA-TSR-StdConv', in_channels=[256, 512, 1024, 2048], out_channels=256, norm_type='BN')
```

This keeps the rest of the experimental pipeline unchanged.

---

### 9. Notes

1. The registry name in the config must exactly match the name used in `@MODELS.register_module(name='...')`.
2. The Python class name and the MMDetection registry name are not required to be identical.
3. Hyphens such as `HCA-TSR-NoTau` are valid in the registry string, but **not** valid as Python class names.
4. For fair ablation experiments, keep the backbone, training schedule, augmentation, optimizer, and evaluation settings unchanged.
5. When comparing results, only change the target ablation factor.

---

### 10. Suggested Experiment Naming

Recommended experiment names:

- `retinanet_hcatsr_no_channel_gate`
- `retinanet_hcatsr_no_detail_bypass`
- `retinanet_hcatsr_no_spatial_gate`
- `retinanet_hcatsr_no_tau`
- `retinanet_hcatsr_fixed_blend_weights`
- `retinanet_hcatsr_stdconv`

This helps maintain consistency between code, configs, logs, and paper tables.

---

## 简体中文

### 1. 文档说明

本仓库包含 **6 个 HCA-TSR 消融 neck 模块**，用于 **MMDetection** 框架下的目标检测实验。  
这些模块主要用于做控制变量实验，分析 HCA-TSR 结构中各个关键子模块的贡献。

---

### 2. 包含的 6 个消融模块

| 消融内容 | 注册名 | Python 颈部类名 | 文件名 | 说明 |
|---|---|---|---|---|
| 去掉 ChannelGate | `HCA-TSR-NoChannelGate` | `HCATSRNoChannelGateNeck` | `HCA_TSR_NoChannelGate.py` | 去掉 HCA 分支中的通道门控 |
| 去掉 DetailBypass | `HCA-TSR-NoDetailBypass` | `HCATSRNoDetailBypassNeck` | `HCA_TSR_NoDetailBypass.py` | 去掉细节增强旁路 |
| 去掉 SpatialGate | `HCA-TSR-NoSpatialGate` | `HCATSRNoSpatialGateNeck` | `HCA_TSR_NoSpatialGate.py` | 去掉 TSR 分支中的空间门控 |
| 去掉 Tau | `HCA-TSR-NoTau` | `HCATSRNoTauNeck` | `HCA_TSR_NoTau.py` | 去掉通道门控中的温度系数 `tau` |
| 固定融合权重 | `HCA-TSR-FixedBlendWeights` | `HCATSRFixedBlendWeightsNeck` | `HCA_TSR_FixedBlendWeights.py` | 将三分支融合权重改为固定常量 |
| 用 StdConv 替换 DW | `HCA-TSR-StdConv` | `HCATSRStdConvNeck` | `HCA_TSR_StdConv.py` | 用标准卷积替代深度可分离卷积 |

---

### 3. 在 MMDetection 中的推荐放置方式

建议把这些文件放到 `mmdet/models/necks/` 目录下，例如：

```text
mmdetection/
├─ mmdet/
│  ├─ models/
│  │  ├─ necks/
│  │  │  ├─ HCA_TSR_NoChannelGate.py
│  │  │  ├─ HCA_TSR_NoDetailBypass.py
│  │  │  ├─ HCA_TSR_NoSpatialGate.py
│  │  │  ├─ HCA_TSR_NoTau.py
│  │  │  ├─ HCA_TSR_FixedBlendWeights.py
│  │  │  ├─ HCA_TSR_StdConv.py
│  │  │  └─ __init__.py
```

然后在 `mmdet/models/necks/__init__.py` 中导入：

```python
from .HCA_TSR_NoChannelGate import HCATSRNoChannelGateNeck
from .HCA_TSR_NoDetailBypass import HCATSRNoDetailBypassNeck
from .HCA_TSR_NoSpatialGate import HCATSRNoSpatialGateNeck
from .HCA_TSR_NoTau import HCATSRNoTauNeck
from .HCA_TSR_FixedBlendWeights import HCATSRFixedBlendWeightsNeck
from .HCA_TSR_StdConv import HCATSRStdConvNeck

__all__ = [
    'HCATSRNoChannelGateNeck',
    'HCATSRNoDetailBypassNeck',
    'HCATSRNoSpatialGateNeck',
    'HCATSRNoTauNeck',
    'HCATSRFixedBlendWeightsNeck',
    'HCATSRStdConvNeck',
]
```

---

### 4. 配置文件里如何调用

这些模块在代码中都是通过：

```python
@MODELS.register_module(name='...')
```

注册到 MMDetection 中的。  
所以在配置文件里，应该写的是 **注册名**，不是 Python 类名。

例如：

```python
neck=dict(
    type='HCA-TSR-NoChannelGate',
    in_channels=[256, 512, 1024, 2048],
    out_channels=256,
    norm_type='BN'
)
```

---

### 5. 六个模块的配置示例

#### 5.1 去掉 ChannelGate

```python
model = dict(
    neck=dict(
        type='HCA-TSR-NoChannelGate',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        norm_type='BN'
    )
)
```

#### 5.2 去掉 DetailBypass

```python
model = dict(
    neck=dict(
        type='HCA-TSR-NoDetailBypass',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        norm_type='BN'
    )
)
```

#### 5.3 去掉 SpatialGate

```python
model = dict(
    neck=dict(
        type='HCA-TSR-NoSpatialGate',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        norm_type='BN'
    )
)
```

#### 5.4 去掉 Tau

```python
model = dict(
    neck=dict(
        type='HCA-TSR-NoTau',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        norm_type='BN'
    )
)
```

#### 5.5 固定融合权重

```python
model = dict(
    neck=dict(
        type='HCA-TSR-FixedBlendWeights',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        norm_type='BN',
        fixed_weights=(0.6, 0.3, 0.1)
    )
)
```

#### 5.6 用 StdConv 替换 DW

```python
model = dict(
    neck=dict(
        type='HCA-TSR-StdConv',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        norm_type='BN'
    )
)
```

---

### 6. 适配的 backbone 说明

这 6 个消融 neck 都遵循 **类似 FPN 的多层输入接口**，适合大多数会输出四层特征图的 backbone，例如：

- ResNet-50 / ResNet-101
- ResNeXt
- 接入 MMDetection 的 EfficientNet
- 其他能输出四层金字塔特征的主干网络

常见写法：

```python
in_channels=[256, 512, 1024, 2048]
out_channels=256
```

如果你的 backbone 输出通道不同，就按实际情况修改 `in_channels`。

---

### 7. 输出特征层说明

这 6 个模块输出统一为：

```python
(P3, P4, P5, P6)
```

对应大致 stride 为：

- `P3`: 8
- `P4`: 16
- `P5`: 32
- `P6`: 64

因此可以直接用于 MMDetection 中很多 one-stage 和 two-stage 检测器。

---

### 8. 与 RetinaNet 联用示例

```python
model = dict(
    type='RetinaNet',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
    ),
    neck=dict(
        type='HCA-TSR-NoSpatialGate',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        norm_type='BN'
    ),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=26,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64]
        ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]
        ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0
        ),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)
    )
)
```

---

### 9. 如何快速切换 6 个消融版本

做消融实验时，最简单的方式就是只改 `type`。

```python
# NoChannelGate
neck=dict(type='HCA-TSR-NoChannelGate', in_channels=[256, 512, 1024, 2048], out_channels=256, norm_type='BN')

# NoDetailBypass
neck=dict(type='HCA-TSR-NoDetailBypass', in_channels=[256, 512, 1024, 2048], out_channels=256, norm_type='BN')

# NoSpatialGate
neck=dict(type='HCA-TSR-NoSpatialGate', in_channels=[256, 512, 1024, 2048], out_channels=256, norm_type='BN')

# NoTau
neck=dict(type='HCA-TSR-NoTau', in_channels=[256, 512, 1024, 2048], out_channels=256, norm_type='BN')

# FixedBlendWeights
neck=dict(type='HCA-TSR-FixedBlendWeights', in_channels=[256, 512, 1024, 2048], out_channels=256, norm_type='BN', fixed_weights=(0.6, 0.3, 0.1))

# StdConv
neck=dict(type='HCA-TSR-StdConv', in_channels=[256, 512, 1024, 2048], out_channels=256, norm_type='BN')
```

这样可以保证除了目标消融因素外，其他实验条件都不变。

---

### 10. 使用注意事项

1. 配置文件中的 `type` 必须和 `@MODELS.register_module(name='...')` 中的注册名完全一致。
2. Python 类名和 MMDetection 注册名可以不同。
3. 像 `HCA-TSR-NoTau` 这种带连字符的名字可以作为注册名，但不能作为 Python 类名。
4. 做公平消融实验时，backbone、训练轮数、数据增强、优化器、评估方式都不要变。
5. 每次只改一个目标因素，实验结论才有说服力。

---

### 11. 建议的实验命名方式

建议使用下面这种命名方式统一管理实验：

- `retinanet_hcatsr_no_channel_gate`
- `retinanet_hcatsr_no_detail_bypass`
- `retinanet_hcatsr_no_spatial_gate`
- `retinanet_hcatsr_no_tau`
- `retinanet_hcatsr_fixed_blend_weights`
- `retinanet_hcatsr_stdconv`

这样代码、配置、日志、论文表格会比较一致。
