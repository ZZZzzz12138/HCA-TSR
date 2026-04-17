import os
import json
from collections import Counter
import matplotlib.pyplot as plt

# 中文字体 & 负号显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def _int(x):
    """安全地将可能为 str/int 的 id 转为 int"""
    try:
        return int(x)
    except Exception:
        return x


def _maybe_remap_zero_based(ann_ids, cat_ids):
    """
    识别并修正 YOLO 风格(0..K-1) 与 COCO 风格(1..K) 的偏移。
    条件：categories 最小 id == 1，annotations 出现 0，且 max(cat)==max(ann)+1
    返回一个映射函数，用于把 annotation 的 id 做 +1 或原样。
    """
    if not ann_ids or not cat_ids:
        return lambda cid: cid

    a_min, a_max = min(ann_ids), max(ann_ids)
    c_min, c_max = min(cat_ids), max(cat_ids)

    if c_min == 1 and a_min == 0 and c_max == a_max + 1:
        return lambda cid: cid + 1
    return lambda cid: cid


def check_image_overlap(train_json, test_json):
    with open(train_json, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(test_json, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    train_imgs = set(img['file_name'] for img in train_data.get('images', []))
    test_imgs  = set(img['file_name'] for img in test_data.get('images', []))

    overlap = train_imgs & test_imgs
    return {
        "train_total": len(train_imgs),
        "test_total": len(test_imgs),
        "overlap_count": len(overlap),
        "overlap_files": list(overlap)
    }


def analyze_bbox_area_distribution(json_path, output_path, dataset_name=''):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    areas = [ann['bbox'][2] * ann['bbox'][3] for ann in data.get('annotations', [])]
    if not areas:
        print(f"⚠️ No annotations found in {dataset_name}")
        return

    # COCO 小/中/大阈值（面积）
    small_count  = sum(a < 32 * 32 for a in areas)
    medium_count = sum(32 * 32 <= a < 96 * 96 for a in areas)
    large_count  = sum(a >= 96 * 96 for a in areas)
    total = len(areas)

    print(f"📊 【{dataset_name} BBox分布统计】")
    print(f"  小目标（<1024）：{small_count}，占比 {small_count / total:.2%}")
    print(f"  中目标（1024~9216）：{medium_count}，占比 {medium_count / total:.2%}")
    print(f"  大目标（≥9216）：{large_count}，占比 {large_count / total:.2%}")

    # 绘图
    plt.figure(figsize=(8, 6))
    plt.hist(areas, bins=30)
    plt.axvline(x=1024, linestyle='--', label='小目标上限 (1024)')
    plt.axvline(x=9216, linestyle='--', label='中目标上限 (9216)')
    plt.title(f"{dataset_name} BBox Area Distribution")
    plt.xlabel("Area (width × height)")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()

    filename = f"{dataset_name.lower()}_bbox_area_distribution.png"
    plt.savefig(os.path.join(output_path, filename))
    plt.close()


def analyze_category_distribution(json_path, output_path, dataset_name=''):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 1) 构建 id->name，统一类型为 int，并给兜底名称
    categories = { _int(cat['id']): str(cat.get('name', f"cat_{cat['id']}"))
                   for cat in data.get('categories', []) }
    cat_ids_defined = set(categories.keys())

    # 2) 统计 annotations 中的类别 id（统一为 int）
    raw_ann_ids = [_int(ann['category_id']) for ann in data.get('annotations', [])]
    ann_ids_set = set(raw_ann_ids)

    # 3) 自动识别并修正 0/1 偏移（若满足典型条件）
    remap = _maybe_remap_zero_based(ann_ids_set, cat_ids_defined)
    remapped_ann_ids = [remap(cid) for cid in raw_ann_ids]
    cat_counter = Counter(remapped_ann_ids)

    # 4) 警告：若仍有 categories 未定义的 id
    missing = sorted(set(cat_counter.keys()) - cat_ids_defined)
    if missing:
        print(f"[WARN] {dataset_name}: annotations 中存在 categories 未定义的类别ID: {missing}")

    # 5) 生成名称与计数（使用 get 防止 KeyError）
    #    这里按数量降序排，便于阅读
    items = sorted(cat_counter.items(), key=lambda kv: kv[1], reverse=True)
    cat_names = [categories.get(cid, f"unknown_{cid}") for cid, _ in items]
    counts    = [cnt for _, cnt in items]

    # 绘图
    plt.figure(figsize=(12, 6))
    plt.bar(cat_names, counts)
    plt.xticks(rotation=90)
    plt.title(f"{dataset_name} 类别分布")
    plt.xlabel("类别")
    plt.ylabel("数量")
    plt.tight_layout()
    filename = f'{dataset_name.lower()}_category_distribution.png'
    plt.savefig(os.path.join(output_path, filename))
    plt.close()

    return dict(zip(cat_names, counts))


def generate_dataset_summary(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    img_count = len(data.get('images', []))
    ann_count = len(data.get('annotations', []))
    avg_ann   = ann_count / img_count if img_count else 0.0

    bbox_areas = [ann['bbox'][2] * ann['bbox'][3] for ann in data.get('annotations', [])]

    # 类别分布：统一类型 + 自动 0/1 对齐 + 兜底
    categories = { _int(cat['id']): str(cat.get('name', f"cat_{cat['id']}"))
                   for cat in data.get('categories', []) }
    cat_ids_defined = set(categories.keys())

    raw_ann_ids = [_int(ann['category_id']) for ann in data.get('annotations', [])]
    ann_ids_set = set(raw_ann_ids)

    remap = _maybe_remap_zero_based(ann_ids_set, cat_ids_defined)
    remapped_ann_ids = [remap(cid) for cid in raw_ann_ids]

    category_counter = Counter(remapped_ann_ids)
    missing = sorted(set(category_counter.keys()) - cat_ids_defined)
    if missing:
        print(f"[WARN] Summary: annotations 中存在 categories 未定义的类别ID: {missing}")

    category_distribution = {
        categories.get(k, f"unknown_{k}"): v
        for k, v in category_counter.items()
    }

    # ✅ 使用 Counter 直接统计所有图片的分辨率（修复原来恒为1的问题）
    res_counter = Counter(f"{img['width']}x{img['height']}" for img in data.get('images', []))

    return {
        "total_images": img_count,
        "total_annotations": ann_count,
        "average_annotations_per_image": round(avg_ann, 2),
        "category_distribution": dict(sorted(category_distribution.items(), key=lambda kv: kv[1], reverse=True)),
        "image_resolutions": dict(sorted(res_counter.items(), key=lambda kv: kv[1], reverse=True)),
        "bbox_area_min": min(bbox_areas) if bbox_areas else None,
        "bbox_area_max": max(bbox_areas) if bbox_areas else None
    }


def run_all_analysis(train_json, test_json, output_path='./analysis_report'):
    os.makedirs(output_path, exist_ok=True)
    print("✅ 开始执行一键分析脚本.")

    # 图像文件名重叠检查
    overlap_result = check_image_overlap(train_json, test_json)

    # BBox 面积分布
    analyze_bbox_area_distribution(train_json, output_path, dataset_name='Train')
    analyze_bbox_area_distribution(test_json,  output_path, dataset_name='Test')

    # 类别分布
    train_cat_distribution = analyze_category_distribution(train_json, output_path, dataset_name='Train')
    test_cat_distribution  = analyze_category_distribution(test_json,  output_path, dataset_name='Test')

    # 汇总
    train_summary = generate_dataset_summary(train_json)
    test_summary  = generate_dataset_summary(test_json)

    summary = {
        "train_summary": train_summary,
        "test_summary": test_summary,
        "overlap_check": overlap_result,
        "train_category_distribution": train_cat_distribution,
        "test_category_distribution": test_cat_distribution
    }

    with open(os.path.join(output_path, 'full_dataset_analysis.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("🎉 所有分析完成，结果保存在：", output_path)


# 示例入口调用（可自行修改路径）
if __name__ == "__main__":
    run_all_analysis(
        train_json=r"D:\Desktop\OBJECT_DETECTION\data\plantsegv3\annotation_train.json",
        test_json=r"D:\Desktop\OBJECT_DETECTION\data\plantsegv3\annotation_val.json",
        output_path='./analysis_report/Plant_Seg'
    )
