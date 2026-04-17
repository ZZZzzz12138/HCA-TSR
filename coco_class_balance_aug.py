"""
按类别补齐目标数量的 COCO 检测数据增强脚本

功能说明：
1. 读取 COCO 格式标注文件
2. 统计各类别目标数量
3. 对样本不足的类别进行图像增强
4. 生成新的图像文件与对应标注
5. 输出增强后的 COCO 标注文件
"""

import os
import json
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import albumentations as A
from collections import defaultdict, Counter

# === 路径配置 ===
root_dir = r"D:/Desktop/data/Casva_leaf disease"
img_dir = os.path.join(root_dir, "train")
json_path = os.path.join(root_dir, "train_annotations.json")
output_json_path = os.path.join(root_dir, "train_annotations_augmented.json")

# === 参数设置 ===
target_per_class = 300  # 每个类别希望补齐到的最小目标数量
max_aug_per_image = 3  # 预留参数：单张图最多增强次数，用于限制重复增强

# === 加载 COCO 标注文件 ===
with open(json_path, "r", encoding="utf-8") as f:
    coco = json.load(f)

cat_id2name = {c["id"]: c["name"] for c in coco["categories"]}
cat_name2id = {v: k for k, v in cat_id2name.items()}

image_id_to_anns = defaultdict(list)
for ann in coco["annotations"]:
    image_id_to_anns[ann["image_id"]].append(ann)

image_id_to_info = {img["id"]: img for img in coco["images"]}
category_counts = Counter([ann["category_id"] for ann in coco["annotations"]])

next_img_id = max(img["id"] for img in coco["images"]) + 1  # 新增图像的起始 ID
next_ann_id = max(ann["id"] for ann in coco["annotations"]) + 1  # 新增标注的起始 ID

# === 定义数据增强策略 ===
augmenter = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.HueSaturationValue(p=0.3),
    A.Rotate(limit=20, p=0.3),
    A.GaussianBlur(p=0.2),
], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))

# === 准备增强结果容器 ===
new_images, new_annotations = [], []

# 逐类别检查是否需要补齐样本数量
for cat_id, count in category_counts.items():
    if count >= target_per_class:
        continue  # 当前类别数量已达标，无需增强

    shortfall = target_per_class - count

    # 找出包含该类别目标的相关图像 ID
    related_image_ids = list({
        ann["image_id"]
        for ann in coco["annotations"]
        if ann["category_id"] == cat_id
    })

    for _ in tqdm(range(shortfall), desc=f"增强类别 {cat_id2name[cat_id]}"):
        base_img_id = random.choice(related_image_ids)
        base_img_info = image_id_to_info[base_img_id]
        base_img_path = os.path.join(img_dir, base_img_info["file_name"])
        # 如果原图不存在，则跳过当前样本
        if not os.path.exists(base_img_path):
            continue

        img = np.array(Image.open(base_img_path).convert("RGB"))
        anns = [a for a in image_id_to_anns[base_img_id] if a["category_id"] == cat_id]
        bboxes = [a["bbox"] for a in anns]
        cat_ids = [a["category_id"] for a in anns]

        # 如果当前类别在该图中没有可用框，则跳过
        if not bboxes:
            continue

        # 增强过程中如果边界框变换失败，则跳过当前样本
        try:
            aug = augmenter(image=img, bboxes=bboxes, category_ids=cat_ids)
        except Exception:
            continue

        # 生成增强后的新文件名
        new_filename = f"{os.path.splitext(base_img_info['file_name'])[0]}_aug_{random.randint(1000, 9999)}.jpg"
        new_img_path = os.path.join(img_dir, new_filename)
        Image.fromarray(aug['image']).save(new_img_path)

        new_images.append({
            "id": next_img_id,
            "file_name": new_filename,
            "width": aug['image'].shape[1],
            "height": aug['image'].shape[0],
        })

        for bbox, cid in zip(aug['bboxes'], aug['category_ids']):
            new_annotations.append({
                "id": next_ann_id,
                "image_id": next_img_id,
                "category_id": cid,
                "bbox": [round(v, 2) for v in bbox],
                "area": round(bbox[2] * bbox[3], 2),
                "iscrowd": 0
            })
            next_ann_id += 1

        next_img_id += 1

# === 保存增强后的标注文件 ===
coco["images"].extend(new_images)
coco["annotations"].extend(new_annotations)
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(coco, f, indent=2)

print(f"✅ 数据增强完成，结果已保存到: {output_json_path}")
