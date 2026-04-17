# -*- coding: utf-8 -*-
import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# 路径配置
# -----------------------------
input_path = 'loss/atss_loss_data.json'
output_dir = 'loss_curve'
output_path = os.path.join(output_dir, 'atss_loss_curve.png')

# -----------------------------
# 加载数据
# -----------------------------
with open(input_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 提取 step 和 epoch
steps = [item['step'] for item in data]
epochs = [item['epoch'] for item in data]

# 自动检测所有与 loss 相关的字段
keys_to_plot = [k for k in data[0].keys() if 'loss' in k.lower()]

# 提取每个字段的值
loss_dict = {k: [item[k] for item in data] for k in keys_to_plot}

# -----------------------------
# 平滑函数
# -----------------------------
def smooth(y, box_pts=15):
    box = np.ones(box_pts) / box_pts
    return np.convolve(y, box, mode='same')

# 平滑处理
loss_smooth_dict = {k: smooth(v) for k, v in loss_dict.items()}

# -----------------------------
# 绘图
# -----------------------------
plt.figure(figsize=(14, 7))  # 设置图像尺寸

# 自动生成不同颜色
colors = matplotlib.colormaps['tab10'].resampled(len(loss_smooth_dict))

linestyles = ['-', '--', ':', '-.', (0, (5, 1))]  # 循环使用线型

for i, (k, v) in enumerate(loss_smooth_dict.items()):
    plt.plot(steps, v,
             label=k,
             color=colors(i),                    # 自动分配颜色
             linestyle=linestyles[i % len(linestyles)],
             linewidth=2 if i==0 else 1.5)

# epoch 竖线标注
shown_epochs = set()
max_loss = max([max(v) for v in loss_smooth_dict.values()])
for i in range(len(steps)):
    ep = epochs[i]
    if ep not in shown_epochs:
        shown_epochs.add(ep)
        plt.axvline(x=steps[i], color='gray', linestyle='--', alpha=0.25)
        plt.text(steps[i], max_loss + 0.002, f'Epoch {ep}', rotation=0, va='bottom', ha='center', fontsize=8)

# 美化图表
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Smoothed Loss Curves by Step with Epoch Markers')
plt.grid(True)
plt.legend(loc='upper right')
plt.tight_layout()

# -----------------------------
# 保存图像
# -----------------------------
os.makedirs(output_dir, exist_ok=True)
plt.savefig(output_path, dpi=300)
print(f"✅ 已保存平滑 loss 曲线图：{output_path}")
