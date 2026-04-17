import os
import json

# 文件路径（修改为实际路径）
annotation_file = r'D:\Desktop\OBJECT_DETECTION\data\Tomato-Village-main\Tomato-Village-main\Variant-c(Object Detection)\annotations\train_coco.json'
image_root = r'D:\Desktop\OBJECT_DETECTION\data\Tomato-Village-main\Tomato-Village-main\Variant-c(Object Detection)\train\images'

# 读取增强后的 COCO 格式标注文件
with open(annotation_file, 'r', encoding='utf-8') as f:
    coco = json.load(f)

# 获取所有增强后的图片文件名
existing_images = {img['file_name']: img['id'] for img in coco['images']}
missing_images = []
for filename, img_id in existing_images.items():
    full_path = os.path.join(image_root, filename)
    if not os.path.exists(full_path):
        missing_images.append(filename)

# 检查增强后的标注
missing_annotations = []
for annotation in coco['annotations']:
    img_id = annotation['image_id']
    img_info = next((img for img in coco['images'] if img['id'] == img_id), None)
    if img_info and img_info['file_name'] not in existing_images:
        missing_annotations.append(img_info['file_name'])

# 输出结果
print(f"总共增强后的标注图片数: {len(coco['images'])}")
print(f"缺失图片数: {len(missing_images)}")
if missing_images:
    print("缺失的图片文件名如下：")
    for name in missing_images:
        print(name)
else:
    print("所有增强后的图片都存在。")

# 检查标注是否丢失
if missing_annotations:
    print(f"以下标注的图片文件在增强后缺失：")
    for name in missing_annotations:
        print(name)
else:
    print("所有增强后的图片都有对应标注。")
