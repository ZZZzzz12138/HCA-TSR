# English | [简体中文](#简体中文)

HCA-TSR is a small MMDetection-oriented toolkit centered on a custom neck module for object detection, together with ablation modules, dataset analysis scripts, annotation checking utilities, data balancing augmentation, and training-curve visualization scripts.

This README is written with **English as the primary language** and includes a Chinese section for switched reading.

---

## English

### 1. Project Overview

This project contains four functional parts:

1. **Main neck module**
   - `HCA_TSR_neck.py`
   - Main custom neck implementation for MMDetection

2. **Ablation modules**
   - `Ablation_Module/`
   - Six controlled ablation neck variants for component-level experiments

3. **Dataset and annotation tools**
   - `Dataset_Quality_Analysis.py`
   - `Check_Annotation_Integrity.py`
   - `coco_class_balance_aug.py`

4. **Training visualization scripts and result assets**
   - `loss_map/`
   - `analysis_report/`

The project is suitable for:
- MMDetection-based object detection research
- Custom neck replacement experiments
- Ablation studies
- COCO-format dataset inspection
- Simple class balancing through augmentation
- Loss and mAP curve plotting

---

### 2. Project Structure

```text
HCA-TSR/
├─ HCA_TSR_neck.py
├─ Check_Annotation_Integrity.py
├─ Dataset_Quality_Analysis.py
├─ coco_class_balance_aug.py
├─ Ablation_Module/
│  ├─ HCA_TSR_NoChannelGate.py
│  ├─ HCA_TSR_NoDetailBypass.py
│  ├─ HCA_TSR_NoSpatialGate.py
│  ├─ HCA_TSR_NoTau.py
│  ├─ HCA_TSR_FixedBlendWeights.py
│  ├─ HCA_TSR_StdConv.py
│  └─ HCA_TSR_Ablation_README.md
├─ analysis_report/
│  ├─ full_dataset_analysis.json
│  ├─ category_distribution.png
│  ├─ bbox_area_distribution.png
│  ├─ train_category_distribution.png
│  ├─ test_category_distribution.png
│  └─ ...
└─ loss_map/
   ├─ loss_print.py
   ├─ map_print.py
   ├─ cut.py
   ├─ 111.py
   ├─ loss/
   ├─ map/
   ├─ loss_curve/
   ├─ map_curve/
   └─ orginal_data/
```

---

### 3. Core Files and Their Roles

#### 3.1 `HCA_TSR_neck.py`

This is the main custom neck file.  
It defines the full **HCA-TSR neck** for MMDetection.

Main purpose:
- Provide a custom multi-scale feature fusion neck
- Replace standard FPN-like necks in MMDetection experiments
- Support object detection ablation and performance comparison

Use this file when:
- You want to run the **full HCA-TSR model**
- You want to compare HCA-TSR against FPN, PAFPN, BiFPN, or other necks

---

#### 3.2 `Ablation_Module/`

This folder contains six ablation variants:

| File | Registry Name | Purpose |
|---|---|---|
| `HCA_TSR_NoChannelGate.py` | `HCA-TSR-NoChannelGate` | Remove channel gating |
| `HCA_TSR_NoDetailBypass.py` | `HCA-TSR-NoDetailBypass` | Remove detail bypass |
| `HCA_TSR_NoSpatialGate.py` | `HCA-TSR-NoSpatialGate` | Remove spatial gating |
| `HCA_TSR_NoTau.py` | `HCA-TSR-NoTau` | Remove temperature coefficient tau |
| `HCA_TSR_FixedBlendWeights.py` | `HCA-TSR-FixedBlendWeights` | Replace learnable fusion weights with fixed weights |
| `HCA_TSR_StdConv.py` | `HCA-TSR-StdConv` | Replace depthwise separable convolution with standard convolution |

Use these files when:
- You are running ablation studies
- You want to isolate the contribution of a specific component
- You want consistent MMDetection experiments with minimal config changes

---

#### 3.3 `Dataset_Quality_Analysis.py`

This script is used to analyze dataset quality, especially for COCO-style detection annotations.

Main functions:
- Category distribution analysis
- Bounding box area distribution analysis
- Train/test image overlap checking
- Export charts and JSON reports

Use this file when:
- You want to inspect class imbalance
- You want to inspect small/medium/large object distribution
- You want to verify train/test split quality

Typical output:
- category distribution figures
- bbox area distribution figures
- analysis report JSON

---

#### 3.4 `Check_Annotation_Integrity.py`

This script checks whether all annotated images actually exist on disk and whether annotations still match image entries.

Use this file when:
- You have augmented the dataset
- You have moved or copied data
- You want to avoid training crashes caused by broken image paths

It is especially useful after running augmentation scripts.

---

#### 3.5 `coco_class_balance_aug.py`

This is a COCO-format class balancing augmentation script.

Main purpose:
- Count target instances per category
- Find underrepresented categories
- Augment related images and bounding boxes
- Save new images
- Generate an updated COCO annotation file

Use this file when:
- Your dataset has severe class imbalance
- You want a simple offline augmentation pipeline before MMDetection training

Important note:
- The script currently uses hard-coded local paths and should be edited before use
- It is an offline preprocessing script, not an MMDetection training hook

---

#### 3.6 `loss_map/`

This folder contains scripts and cached data for plotting training metrics.

Typical content:
- loss JSON files
- mAP JSON files
- loss curve plots
- mAP curve plots
- raw log-exported JSON files

Use this folder when:
- You want to visualize training dynamics
- You want to compare models across runs
- You want figures for reports or papers

---

### 4. How to Use This Project with MMDetection

This project is not a full standalone training repository.  
It is best understood as a **custom module and analysis toolkit** intended to be integrated into an existing **MMDetection** codebase.

The typical workflow is:

1. Put the neck files into MMDetection
2. Register the modules
3. Update the config file
4. Prepare your dataset
5. Train and evaluate
6. Use the analysis scripts for reporting

---

### 5. Recommended MMDetection Integration

#### 5.1 Copy the main neck file

Copy:

```text
HCA_TSR_neck.py
```

into:

```text
mmdetection/mmdet/models/necks/
```

If you need ablation experiments, also copy the six files under:

```text
Ablation_Module/
```

into the same MMDetection neck directory, or into your custom module directory.

---

#### 5.2 Edit `__init__.py`

Add imports in:

```python
mmdet/models/necks/__init__.py
```

Example:

```python
from .HCA_TSR_neck import HCATSRNeck
from .HCA_TSR_NoChannelGate import HCATSRNoChannelGateNeck
from .HCA_TSR_NoDetailBypass import HCATSRNoDetailBypassNeck
from .HCA_TSR_NoSpatialGate import HCATSRNoSpatialGateNeck
from .HCA_TSR_NoTau import HCATSRNoTauNeck
from .HCA_TSR_FixedBlendWeights import HCATSRFixedBlendWeightsNeck
from .HCA_TSR_StdConv import HCATSRStdConvNeck
```

Then add them to `__all__` if needed.

---

#### 5.3 Use the registry name in config

MMDetection uses the module registry name, not the Python class name.

Example for the full model:

```python
model = dict(
    neck=dict(
        type='HCA-TSR',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        norm_type='BN'
    )
)
```

Example for an ablation model:

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

---

### 6. Example MMDetection Config Snippets

#### 6.1 Full HCA-TSR

```python
neck=dict(
    type='HCA-TSR',
    in_channels=[256, 512, 1024, 2048],
    out_channels=256,
    norm_type='BN'
)
```

#### 6.2 NoChannelGate

```python
neck=dict(
    type='HCA-TSR-NoChannelGate',
    in_channels=[256, 512, 1024, 2048],
    out_channels=256,
    norm_type='BN'
)
```

#### 6.3 NoDetailBypass

```python
neck=dict(
    type='HCA-TSR-NoDetailBypass',
    in_channels=[256, 512, 1024, 2048],
    out_channels=256,
    norm_type='BN'
)
```

#### 6.4 NoSpatialGate

```python
neck=dict(
    type='HCA-TSR-NoSpatialGate',
    in_channels=[256, 512, 1024, 2048],
    out_channels=256,
    norm_type='BN'
)
```

#### 6.5 NoTau

```python
neck=dict(
    type='HCA-TSR-NoTau',
    in_channels=[256, 512, 1024, 2048],
    out_channels=256,
    norm_type='BN'
)
```

#### 6.6 FixedBlendWeights

```python
neck=dict(
    type='HCA-TSR-FixedBlendWeights',
    in_channels=[256, 512, 1024, 2048],
    out_channels=256,
    norm_type='BN',
    fixed_weights=(0.6, 0.3, 0.1)
)
```

#### 6.7 StdConv

```python
neck=dict(
    type='HCA-TSR-StdConv',
    in_channels=[256, 512, 1024, 2048],
    out_channels=256,
    norm_type='BN'
)
```

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
        type='HCA-TSR',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        norm_type='BN'
    ),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=26,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256
    )
)
```

---

### 8. How to Use the Dataset Scripts

#### 8.1 Analyze dataset quality

Edit dataset paths inside `Dataset_Quality_Analysis.py`, then run:

```bash
python Dataset_Quality_Analysis.py
```

Use it before training to answer:
- Is the dataset imbalanced
- Are there many small objects
- Does train/test overlap exist
- Are the categories distributed reasonably

---

#### 8.2 Check annotation integrity

Edit paths in `Check_Annotation_Integrity.py`, then run:

```bash
python Check_Annotation_Integrity.py
```

Use it after:
- offline augmentation
- file migration
- annotation regeneration

---

#### 8.3 Balance categories with augmentation

Edit paths in `coco_class_balance_aug.py`, then run:

```bash
python coco_class_balance_aug.py
```

Use it before training if:
- some categories are severely underrepresented
- you want to produce an augmented annotation JSON file

Recommended practice:
- back up original annotations first
- inspect the new output JSON
- run `Check_Annotation_Integrity.py` after augmentation

---

### 9. How to Use the Visualization Scripts

Inside `loss_map/`, you can use the scripts to process stored JSON logs and plot:
- training loss curves
- mAP curves
- model comparison charts

Because these scripts are lightweight utilities, you should first inspect the expected JSON file paths and formats before running them.

Typical usage:

```bash
python loss_map/loss_print.py
python loss_map/map_print.py
```

Use these scripts after training to create:
- figures for papers
- figures for internal reports
- quick comparisons between ablation runs

---

### 10. Suggested Workflow for This Project

A practical order of use is:

1. **Inspect your dataset**
   - `Dataset_Quality_Analysis.py`

2. **Balance classes if necessary**
   - `coco_class_balance_aug.py`

3. **Check the integrity of the updated annotations**
   - `Check_Annotation_Integrity.py`

4. **Integrate HCA-TSR into MMDetection**
   - `HCA_TSR_neck.py`

5. **Run full-model training**
   - with `type='HCA-TSR'`

6. **Run ablation experiments**
   - switch only the `type` field in the config

7. **Visualize loss and mAP curves**
   - `loss_map/`

8. **Collect figures and reports**
   - `analysis_report/`

---

### 11. Important Notes

1. This project uses **COCO-format detection annotations** in multiple scripts.
2. Several scripts contain **hard-coded Windows paths** and should be edited before use.
3. The data-processing scripts are **offline tools**, not integrated MMDetection pipeline components.
4. The neck files are intended for **MMDetection integration**, not for direct standalone training.
5. For fair ablation experiments, keep the backbone, training schedule, optimizer, and data split unchanged.

---

### 12. Recommended File Naming for Experiments

Suggested experiment names:

- `retinanet_hcatsr`
- `retinanet_hcatsr_no_channel_gate`
- `retinanet_hcatsr_no_detail_bypass`
- `retinanet_hcatsr_no_spatial_gate`
- `retinanet_hcatsr_no_tau`
- `retinanet_hcatsr_fixed_blend_weights`
- `retinanet_hcatsr_stdconv`

This helps keep configs, logs, figures, and paper tables consistent.

---

## 简体中文

### 1. 项目说明

HCA-TSR 是一个围绕 **MMDetection 自定义 neck 模块**构建的小型工具项目，同时包含：
- 主 neck 模块
- 6 个消融模块
- 数据集分析脚本
- 标注完整性检查脚本
- 类别平衡增强脚本
- loss / mAP 可视化脚本

它更适合被看成一个 **基于 MMDetection 的研究工具包**，而不是一个完整独立训练框架。

---

### 2. 项目主要内容

这个项目大致分成 4 类文件：

1. **主 neck 文件**
   - `HCA_TSR_neck.py`

2. **消融模块**
   - `Ablation_Module/`

3. **数据处理与检查脚本**
   - `Dataset_Quality_Analysis.py`
   - `Check_Annotation_Integrity.py`
   - `coco_class_balance_aug.py`

4. **训练结果可视化与分析结果**
   - `loss_map/`
   - `analysis_report/`

---

### 3. 各文件的作用

#### 3.1 `HCA_TSR_neck.py`

这是主模型使用的 neck 文件，用于在 MMDetection 中替换标准 FPN 类结构。

适用场景：
- 你要跑完整的 HCA-TSR 模型
- 你要和 FPN、PAFPN、BiFPN 等 neck 做对比实验

---

#### 3.2 `Ablation_Module/`

该目录下包含 6 个消融模块，用于控制变量实验。

| 文件 | 注册名 | 作用 |
|---|---|---|
| `HCA_TSR_NoChannelGate.py` | `HCA-TSR-NoChannelGate` | 去掉通道门控 |
| `HCA_TSR_NoDetailBypass.py` | `HCA-TSR-NoDetailBypass` | 去掉细节旁路 |
| `HCA_TSR_NoSpatialGate.py` | `HCA-TSR-NoSpatialGate` | 去掉空间门控 |
| `HCA_TSR_NoTau.py` | `HCA-TSR-NoTau` | 去掉温度系数 tau |
| `HCA_TSR_FixedBlendWeights.py` | `HCA-TSR-FixedBlendWeights` | 固定融合权重 |
| `HCA_TSR_StdConv.py` | `HCA-TSR-StdConv` | 用标准卷积替换深度可分离卷积 |

---

#### 3.3 `Dataset_Quality_Analysis.py`

这个脚本用于做数据集质量分析，主要分析 COCO 格式目标检测数据。

它可以做：
- 类别分布统计
- bbox 面积分布统计
- train/test 图像重叠检查
- 导出图表与 JSON 分析结果

适用场景：
- 训练前检查类别是否严重不平衡
- 看数据集里小目标是否很多
- 检查训练集和测试集是否有重复图像

---

#### 3.4 `Check_Annotation_Integrity.py`

这个脚本用于检查：
- 标注中的图片是否真实存在
- 标注和图片索引是否一致

特别适合：
- 数据增强后检查
- 数据迁移后检查
- 重新生成标注文件后检查

---

#### 3.5 `coco_class_balance_aug.py`

这个脚本用于对 COCO 数据集做按类别补齐的数据增强。

主要逻辑：
- 统计每一类目标数量
- 找出样本不足的类别
- 对相关图像做增强
- 保存新图像和新标注
- 输出新的 COCO 标注文件

适用场景：
- 某些类别样本太少
- 你希望训练前先做离线增强补齐类别数量

注意：
- 这个脚本目前是本地路径硬编码，使用前要改路径
- 它是离线预处理工具，不是 MMDetection 的在线增强模块

---

#### 3.6 `loss_map/`

这个目录用于保存和绘制训练过程中的指标曲线。

一般包括：
- loss JSON
- mAP JSON
- loss 曲线图
- mAP 曲线图
- 原始训练日志导出的 JSON

适用场景：
- 比较不同模型训练曲线
- 论文中画图
- 汇报中展示训练过程

---

### 4. 如何结合 MMDetection 使用

这个项目最合理的用法不是单独运行，而是把它接入现有的 **MMDetection 工程** 中。

推荐流程是：

1. 把 neck 文件复制到 MMDetection
2. 注册模块
3. 修改配置文件
4. 准备数据集
5. 训练和评估
6. 用分析脚本输出结果图表

---

### 5. 接入 MMDetection 的推荐方式

#### 5.1 复制主 neck 文件

把：

```text
HCA_TSR_neck.py
```

复制到：

```text
mmdetection/mmdet/models/necks/
```

如果要做消融实验，也把 `Ablation_Module/` 下的 6 个文件一起复制进去。

---

#### 5.2 修改 `__init__.py`

在：

```python
mmdet/models/necks/__init__.py
```

中增加导入，例如：

```python
from .HCA_TSR_neck import HCATSRNeck
from .HCA_TSR_NoChannelGate import HCATSRNoChannelGateNeck
from .HCA_TSR_NoDetailBypass import HCATSRNoDetailBypassNeck
from .HCA_TSR_NoSpatialGate import HCATSRNoSpatialGateNeck
from .HCA_TSR_NoTau import HCATSRNoTauNeck
from .HCA_TSR_FixedBlendWeights import HCATSRFixedBlendWeightsNeck
from .HCA_TSR_StdConv import HCATSRStdConvNeck
```

---

#### 5.3 配置文件里使用注册名

在 MMDetection 配置文件里，写的是注册名，不是 Python 类名。

完整模型示例：

```python
model = dict(
    neck=dict(
        type='HCA-TSR',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        norm_type='BN'
    )
)
```

消融模型示例：

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

---

### 6. 配置文件示例

#### 6.1 完整 HCA-TSR

```python
neck=dict(
    type='HCA-TSR',
    in_channels=[256, 512, 1024, 2048],
    out_channels=256,
    norm_type='BN'
)
```

#### 6.2 6 个消融模块

```python
neck=dict(type='HCA-TSR-NoChannelGate', in_channels=[256, 512, 1024, 2048], out_channels=256, norm_type='BN')
neck=dict(type='HCA-TSR-NoDetailBypass', in_channels=[256, 512, 1024, 2048], out_channels=256, norm_type='BN')
neck=dict(type='HCA-TSR-NoSpatialGate', in_channels=[256, 512, 1024, 2048], out_channels=256, norm_type='BN')
neck=dict(type='HCA-TSR-NoTau', in_channels=[256, 512, 1024, 2048], out_channels=256, norm_type='BN')
neck=dict(type='HCA-TSR-FixedBlendWeights', in_channels=[256, 512, 1024, 2048], out_channels=256, norm_type='BN', fixed_weights=(0.6, 0.3, 0.1))
neck=dict(type='HCA-TSR-StdConv', in_channels=[256, 512, 1024, 2048], out_channels=256, norm_type='BN')
```

---

### 7. 数据脚本怎么用

#### 7.1 数据集质量分析

先修改 `Dataset_Quality_Analysis.py` 里的路径，再运行：

```bash
python Dataset_Quality_Analysis.py
```

适合在训练前判断：
- 类别是否失衡
- 小目标是否占比过高
- 训练集和测试集是否重复

---

#### 7.2 标注完整性检查

先修改 `Check_Annotation_Integrity.py` 里的路径，再运行：

```bash
python Check_Annotation_Integrity.py
```

适合在：
- 数据增强后
- 文件迁移后
- 新标注生成后

做一次检查，避免训练时报错。

---

#### 7.3 类别补齐增强

先修改 `coco_class_balance_aug.py` 里的路径，再运行：

```bash
python coco_class_balance_aug.py
```

建议流程：
1. 先备份原始标注
2. 跑增强脚本
3. 再跑完整性检查脚本
4. 最后再喂给 MMDetection 训练

---

### 8. 可视化脚本怎么用

`loss_map/` 目录下的脚本主要是画：
- loss 曲线
- mAP 曲线
- 多模型对比图

常见运行方式：

```bash
python loss_map/loss_print.py
python loss_map/map_print.py
```

适合在训练后生成：
- 论文图片
- 实验汇报图片
- 多组消融结果对比图

---

### 9. 推荐使用顺序

建议你按下面这个顺序用这个项目：

1. 用 `Dataset_Quality_Analysis.py` 看数据集质量
2. 如果类别失衡严重，用 `coco_class_balance_aug.py` 做补齐增强
3. 用 `Check_Annotation_Integrity.py` 检查增强后标注是否正常
4. 把 `HCA_TSR_neck.py` 接入 MMDetection
5. 先跑完整模型
6. 再切换 6 个消融模块做实验
7. 用 `loss_map/` 画训练曲线
8. 用 `analysis_report/` 和曲线图整理论文材料

---

### 10. 注意事项

1. 多个脚本都默认基于 **COCO 格式检测标注**。
2. 项目里有些脚本使用了 **Windows 本地硬编码路径**，运行前必须改。
3. 数据处理脚本是 **离线工具**，不是 MMDetection 训练流程里的在线模块。
4. neck 文件是给 **MMDetection 接入** 用的，不是单独训练脚本。
5. 做消融实验时，尽量保证 backbone、训练轮数、优化器、数据划分不变。

---

### 11. 推荐实验命名

建议实验命名统一成：

- `retinanet_hcatsr`
- `retinanet_hcatsr_no_channel_gate`
- `retinanet_hcatsr_no_detail_bypass`
- `retinanet_hcatsr_no_spatial_gate`
- `retinanet_hcatsr_no_tau`
- `retinanet_hcatsr_fixed_blend_weights`
- `retinanet_hcatsr_stdconv`

这样配置文件、日志、图片、论文表格都会更整齐。
