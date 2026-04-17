import json
import os
import matplotlib.pyplot as plt

# ===== 路径设置 =====
input_path = 'map/mpcaplus_map_data.json'
output_dir = 'map_curve'
output_path = os.path.join(output_dir, 'mpcaplus_map_curve.png')

# ===== 加载 mAP 数据 =====
with open(input_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# ===== 提取绘图数据 =====
steps = [item['step'] for item in data]
map_all = [item['coco/bbox_mAP'] for item in data]
map_50 = [item['coco/bbox_mAP_50'] for item in data]
map_75 = [item['coco/bbox_mAP_75'] for item in data]

# ===== 绘图 =====
plt.figure(figsize=(12, 6))
plt.plot(steps, map_all, label='mAP', linewidth=2)
plt.plot(steps, map_50, label='mAP@50', linestyle='--')
plt.plot(steps, map_75, label='mAP@75', linestyle=':')

# ===== 美化图形 =====
plt.xlabel('Step')
plt.ylabel('mAP')
plt.title('mAP Metrics over Steps')
plt.grid(True)
plt.legend(loc='lower right')
plt.tight_layout()

# ===== 保存图像 =====
os.makedirs(output_dir, exist_ok=True)
plt.savefig(output_path, dpi=300)
print(f"✅ mAP 曲线图已保存到: {output_path}")
