import cv2
import os
import numpy as np
from pathlib import Path
import random
import xml.etree.ElementTree  as ET


def visualize_yolo_labels(
        images_dir: str,
        labels_dir: str,
        class_names_path: str,
        output_dir: str,
        thickness: int = 2
) -> None:
    """
    可视化YOLO格式标签（支持标准YOLO目录结构）

    参数:
        images_dir: YOLO格式图片根目录（需包含train/val子目录）
        labels_dir: YOLO格式标签根目录（需包含train/val子目录）
        class_names_path: 类别名称文件路径（每行一个类别名称）
        output_dir: 可视化结果保存根目录（自动创建train/val子目录）
        thickness: 边界框线宽
    """
    # 初始化目录路径
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    output_path = Path(output_dir)

    # 创建输出子目录
    (output_path / 'train').mkdir(parents=True, exist_ok=True)
    (output_path / 'val').mkdir(parents=True, exist_ok=True)

    # 读取类别名称
    class_names = []
    if class_names_path and Path(class_names_path).exists():
        with open(class_names_path, 'r', encoding='utf-8') as f:
            class_names = [line.strip() for line in f.readlines() if line.strip()]

            # 生成颜色映射（BGR格式）

    def generate_colors(num_colors: int) -> list:
        colors = []
        for i in range(num_colors):
            hue = i * (180 / max(1, num_colors - 1))  # OpenCV中Hue范围0-180
            hsv_color = np.array([[[hue, 255, 255]]], dtype=np.uint8)
            bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
            colors.append(tuple(map(int, bgr_color)))
        return colors

    colors = generate_colors(len(class_names)) if class_names else generate_colors(12)

    # 处理train/val子目录
    for data_type in ['train', 'val']:
        # 检查输入目录是否存在
        if not (images_path / data_type).exists():
            print(f"警告：图片{data_type}目录不存在 - {images_path / data_type}")
            continue
        if not (labels_path / data_type).exists():
            print(f"警告：标签{data_type}目录不存在 - {labels_path / data_type}")
            continue

            # 遍历图片目录
        for img_path in (images_path / data_type).glob("*"):
            if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                continue

                # 读取图片
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"警告：无法读取图片 {img_path}")
                continue

            h, w = img.shape[:2]
            label_path = labels_path / data_type / f"{img_path.stem}.txt"

            # 检查标签文件
            if not label_path.exists():
                print(f"警告：未找到标签文件 {label_path}")
                continue

                # 绘制标签
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue

                        # 解析YOLO格式
                    class_id, x_center, y_center, box_w, box_h = map(float, parts)
                    class_id = int(class_id)

                    # 转换为绝对坐标
                    x_center *= w
                    y_center *= h
                    box_w *= w
                    box_h *= h
                    x1 = int(x_center - box_w / 2)
                    y1 = int(y_center - box_h / 2)
                    x2 = int(x_center + box_w / 2)
                    y2 = int(y_center + box_h / 2)

                    # 绘制边界框和标签
                    color = colors[class_id % len(colors)]
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

                    label = class_names[class_id] if class_names and class_id < len(class_names) else str(class_id)
                    cv2.putText(
                        img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2
                    )

            # 保存结果到对应子目录
            output_file = output_path / data_type / f"vis_{img_path.name}"
            cv2.imwrite(str(output_file), img)
            print(f"已生成{data_type}集可视化结果: {output_file}")


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pycocotools.coco import COCO
from typing import List, Dict, Optional


def visualize_coco_full(
        coco_train_path: str,
        coco_val_path: str,
        coco_annotations_path: str,
        output_vis_path: str,
        class_names: Optional[List[str]] = None,
        dataset_types: List[str] = ['train', 'val'],
        show_bbox: bool = True,
        show_mask: bool = False,
        show_keypoints: bool = False,
        show_crowd: bool = False
) -> Dict[str, str]:
    """
    COCO全量可视化函数

    参数:
        coco_train_path: COCO训练集图片路径 (e.g. 'datasets/coco/train2017')
        coco_val_path: COCO验证集图片路径 (e.g. 'datasets/coco/val2017')
        coco_annotations_path: COCO标注目录 (包含instances_train.json 和instances_val.json)
        output_vis_path: 可视化输出根目录 (自动创建train/val子目录)
        class_names: 可选自定义类别名称列表
        dataset_types: 要处理的数据集类型列表 ['train', 'val']
        show_bbox: 是否显示边界框 (默认True)
        show_mask: 是否显示分割掩码 (默认False)
        show_keypoints: 是否显示关键点 (默认False)
        show_crowd: 是否显示crowd标注 (默认False)

    返回:
        Dict[str, str]: 各数据集可视化结果保存路径 {'train': path1, 'val': path2}
    """
    # 初始化路径配置
    coco_paths = {
        'train': {'img': coco_train_path, 'ann': os.path.join(coco_annotations_path, 'instances_train.json')},
        'val': {'img': coco_val_path, 'ann': os.path.join(coco_annotations_path, 'instances_val.json')}
    }

    # 创建输出目录
    os.makedirs(os.path.join(output_vis_path, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_vis_path, 'val'), exist_ok=True)

    # 初始化COCO API
    coco_apis = {
        dtype: COCO(coco_paths[dtype]['ann'])
        for dtype in dataset_types
    }

    # 获取类别信息
    if class_names is None:
        class_names = [
            cat['name'] for cat in coco_apis['train'].loadCats(
                coco_apis['train'].getCatIds()
            )
        ]
    color_map = plt.cm.get_cmap('tab20', len(class_names))

    def get_color(cat_id: int) -> tuple:
        """获取归一化颜色值 (0-1范围)"""
        return tuple(color_map(cat_id % len(class_names))[:3])

    result_paths = {}

    for dtype in dataset_types:
        print(f"\n{'=' * 40}\nProcessing {dtype} set...\n{'=' * 40}")
        coco = coco_apis[dtype]
        output_dir = os.path.join(output_vis_path, dtype)

        # 处理每张图片
        for img_info in tqdm(coco.loadImgs(coco.getImgIds()),
                             desc=f"Visualizing {dtype} images"):
            try:
                # 加载图像
                img_path = os.path.join(coco_paths[dtype]['img'], img_info['file_name'])
                img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

                # 创建画布
                plt.figure(figsize=(16, 10))
                plt.imshow(img)
                ax = plt.gca()

                # 处理标注
                for ann in coco.loadAnns(coco.getAnnIds(imgIds=img_info['id'])):
                    if not show_crowd and ann.get('iscrowd', 0):
                        continue

                    color = get_color(ann['category_id'])
                    cat = coco.loadCats(ann['category_id'])[0]

                    # 绘制边界框
                    if show_bbox and 'bbox' in ann:
                        bbox = ann['bbox']
                        rect = plt.Rectangle(
                            (bbox[0], bbox[1]), bbox[2], bbox[3],
                            fill=False, edgecolor=color, linewidth=2
                        )
                        ax.add_patch(rect)
                        ax.text(
                            bbox[0], bbox[1] - 10,
                            f"{cat['name']} {ann.get('score', 1.0):.2f}" if 'score' in ann else cat['name'],
                            color='white', fontsize=12, bbox=dict(facecolor=color, alpha=0.8)
                        )

                    # 绘制分割掩码
                    if show_mask and 'segmentation' in ann:
                        if isinstance(ann['segmentation'], list):
                            for seg in ann['segmentation']:
                                poly = np.array(seg).reshape((-1, 2))
                                polygon = plt.Polygon(poly, fill=True, color=color, alpha=0.4)
                                ax.add_patch(polygon)

                    # 绘制关键点
                    if show_keypoints and 'keypoints' in ann:
                        kpts = np.array(ann['keypoints']).reshape(-1, 3)
                        plt.scatter(
                            kpts[kpts[:, 2] > 0][:, 0],
                            kpts[kpts[:, 2] > 0][:, 1],
                            color=color, s=80, marker='o', edgecolors='white'
                        )

                plt.axis('off')
                plt.tight_layout()

                # 保存结果
                output_file = os.path.join(
                    output_dir,
                    f"{os.path.splitext(img_info['file_name'])[0]}_cocovis.jpg"
                )
                plt.savefig(output_file, bbox_inches='tight', dpi=120)
                plt.close()

            except Exception as e:
                print(f"\nError processing {img_info['file_name']}: {str(e)}")
                continue

        result_paths[dtype] = output_dir
        print(f"\n{dtype} set visualization saved to: {output_dir}")

    return result_paths

import colorsys

def generate_distinct_colors(n, seed=42):
    """
    生成N种视觉区分度高的颜色（HSV色彩空间均匀采样）
    参数：
    - n: 需要生成的颜色数量
    - seed: 随机种子（保证颜色一致性）
    """
    random.seed(seed)
    hues = [i/n for i in range(n)]
    random.shuffle(hues)   # 打乱色调顺序
    colors = []
    for h in hues:
        # 固定饱和度和亮度（可调整）
        s = 0.8 + random.random()*0.2   # 饱和度 0.8-1.0
        v = 0.7 + random.random()*0.2   # 亮度 0.7-0.9
        rgb = colorsys.hsv_to_rgb(h,  s, v)
        colors.append((int(rgb[0]*255),  int(rgb[1]*255), int(rgb[2]*255)))
    return colors

def visualize_voc_dataset(
        VOC_AnnotationsPath: str,
        VOC_JPEGImagesPath: str,
        VOC_ImageSetsPath: str,
        OutputVisPath: str,
        show_progress: bool = True
):
    """
    可视化VOC格式数据集（2025年11月4日更新版）

    参数说明：
    - VOC_AnnotationsPath: VOC标注文件路径（Annotations目录）
    - VOC_JPEGImagesPath: VOC图像文件路径（JPEGImages目录）
    - VOC_ImageSetsPath: 数据集划分文件路径（ImageSets/Main目录）
    - OutputVisPath: 可视化结果输出根目录
    - show_progress: 是否显示进度条
    """
    # 创建输出目录
    os.makedirs(os.path.join(OutputVisPath, 'train'), exist_ok=True)
    os.makedirs(os.path.join(OutputVisPath, 'val'), exist_ok=True)

    # 自动发现所有类别（避免硬编码）
    class_names = set()
    for xml_file in os.listdir(VOC_AnnotationsPath):
        if xml_file.endswith('.xml'):
            tree = ET.parse(os.path.join(VOC_AnnotationsPath, xml_file))
            for obj in tree.findall('object'):
                class_names.add(obj.find('name').text)

    # 动态生成颜色
    colors = generate_distinct_colors(len(class_names))
    color_map = {name: colors[i] for i, name in enumerate(sorted(class_names))}

    # 处理每个划分集
    for split in ['train', 'val']:
        print(f"\n🔍 正在处理 {split} 数据集...")
        set_file = os.path.join(VOC_ImageSetsPath, 'Main', f'{split}.txt')

        if not os.path.exists(set_file):
            print(f"⚠️ 跳过 {split} - 划分文件不存在: {set_file}")
            continue

            # 读取样本列表
        with open(set_file, 'r') as f:
            samples = [line.strip() for line in f if line.strip()]

            # 进度条设置
        iter_samples = tqdm(samples) if show_progress else samples

        for sample in iter_samples:
            # 加载图像和标注
            img_path = os.path.join(VOC_JPEGImagesPath, f'{sample}.jpg')
            xml_path = os.path.join(VOC_AnnotationsPath, f'{sample}.xml')

            if not os.path.exists(img_path) or not os.path.exists(xml_path):
                continue

            # 解析XML标注
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # 绘制标注
            img = cv2.imread(img_path)
            for obj in root.findall('object'):
                name = obj.find('name').text
                bbox = obj.find('bndbox')
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)

                # 随机颜色分配（基于类别名称哈希）
                color = color_map[name]

                # 绘制边界框和标签
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
                cv2.putText(img, name, (xmin, ymin - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # 保存可视化结果
            output_path = os.path.join(OutputVisPath, split, f'{sample}_vocvis.jpg')
            cv2.imwrite(output_path, img)

    print("\n🎉 可视化完成！输出结构：")
    print(f"└── {OutputVisPath}")
    print(f"    ├── train/  # 训练集可视化结果")
    print(f"     └── val/    # 验证集可视化结果")

# 使用示例（修改为你的实际路径）
if __name__ == "__main__":
    #############     Yolo      ###############
    # 设置你的resize后图片和标签路径
    resized_images = r"E:\pycharm\DataTransformer\data\images4"  # resize后的图片目录
    adjusted_labels = r"E:\pycharm\DataTransformer\data\labels4"  # 调整后的标签目录

    # 类别名称文件路径（每行一个类别名称）
    class_names_path = r"E:\pycharm\DataTransformer\data\class.txt"  # 修改为你的实际路径

    # 运行可视化
    visualize_yolo_labels(
        images_dir=resized_images,
        labels_dir=adjusted_labels,
        class_names_path=class_names_path,
        output_dir=r"E:\pycharm\DataTransformer\data\Vis"
    )



    #############     COCO      ###############
    # result_paths = visualize_coco_full(
    #     coco_train_path=r'E:\pycharm\DataTransformer\data\coco\train',
    #     coco_val_path=r'E:\pycharm\DataTransformer\data\coco\val',
    #     coco_annotations_path=r'E:\pycharm\DataTransformer\data\coco\annotations',
    #     output_vis_path=r'E:\pycharm\DataTransformer\data\coco\vis',
    #     show_mask=True  # 启用分割掩码可视化
    # )



    # #############     VOC      ###############
    # # 配置路径（根据实际修改）
    # VOC_AnnotationsPath = r'E:\pycharm\DataTransformer\data\VOC\Annotations'
    # VOC_JPEGImagesPath = r'E:\pycharm\DataTransformer\data\VOC\JPEGImages'
    # VOC_ImageSetsPath = r'E:\pycharm\DataTransformer\data\VOC\ImageSets'
    # OutputVisPath = r'E:\pycharm\DataTransformer\data\VOC\visualization'
    #
    # # 执行可视化
    # visualize_voc_dataset(
    #     VOC_AnnotationsPath=VOC_AnnotationsPath,
    #     VOC_JPEGImagesPath=VOC_JPEGImagesPath,
    #     VOC_ImageSetsPath=VOC_ImageSetsPath,
    #     OutputVisPath=OutputVisPath
    # )
    #
