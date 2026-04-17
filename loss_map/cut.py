import json
import os
import pandas as pd

# 路径设置（注意相对路径）
input_path = 'orginal_data/20250814_094302.json'
loss_output_path = 'loss/mpcaplus_loss_data.json'
map_output_path = 'map/mpcaplus_map_data.json'

# 加载 NDJSON 数据（每行一个 JSON 对象）
with open(input_path, 'r', encoding='utf-8') as f:
    raw_data = [json.loads(line) for line in f if line.strip()]

# 拆分
loss_data = [item for item in raw_data if 'loss' in item]
map_data = [item for item in raw_data if 'coco/bbox_mAP' in item]

# 确保目录存在
os.makedirs(os.path.dirname(loss_output_path), exist_ok=True)
os.makedirs(os.path.dirname(map_output_path), exist_ok=True)

# 保存文件
with open(loss_output_path, 'w', encoding='utf-8') as f1:
    json.dump(loss_data, f1, indent=2)

with open(map_output_path, 'w', encoding='utf-8') as f2:
    json.dump(map_data, f2, indent=2)

print("✅ 拆分完成，结果已写入 loss/ 和 map/ 目录。")
