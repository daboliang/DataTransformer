import json
import os
import cv2
from typing import List, Dict, Optional
import shutil


def load_yolo_classes(class_path: str) -> List[Dict]:
    """
    从YOLO格式的classes.txt 加载类别信息
    :param class_path: classes.txt 文件路径
    :return: COCO格式的categories列表
    """
    categories = []
    try:
        with open(class_path, 'r', encoding='utf-8') as f:
            for class_id, class_name in enumerate(f.readlines(), start=1):
                categories.append({
                    "id": class_id,
                    "name": class_name.strip(),
                    "supercategory": "object"  # 默认超类
                })
    except FileNotFoundError:
        print(f"Warning: Class file not found at {class_path}, using default categories")
        # 默认类别（示例）
        categories = [
            {"id": 1, "name": "class0", "supercategory": "object"},
            {"id": 2, "name": "class1", "supercategory": "object"}
        ]
    return categories


def init_coco_structure(class_path: Optional[str] = None) -> Dict:
    """初始化COCO数据结构模板"""
    # 动态加载类别（优先使用传入的class_path）
    categories = load_yolo_classes(class_path) if class_path else load_yolo_classes("classes.txt")

    return {
        'info': {
            "year": 2025,
            "version": '1.0',
            "date_created": "2025-11-03"  # 更新为当前日期
        },
        'licenses': [{
            "id": 1,
            "name": "null",
            "url": "null",
        }],
        'categories': categories,
        'images': [],
        'annotations': []
    }


def convert_yolo_to_coco(image_path, label_path):
    """
    YOLO格式转COCO格式的核心转换函数
    :param image_path: YOLO格式图片路径
    :param label_path: YOLO格式标签路径
    :return: (images列表, annotations列表)
    """
    images = []
    annotations = []

    for index, img_file in enumerate(os.listdir(image_path)):
        if not img_file.endswith('.jpg'):
            continue

            # 处理图像信息
        img = cv2.imread(os.path.join(image_path, img_file))
        height, width = img.shape[:2]
        image_info = {
            'id': index,
            'file_name': img_file,
            'width': width,
            'height': height
        }
        images.append(image_info)

        # 处理标注信息
        label_file = os.path.join(label_path, img_file.replace('.jpg', '.txt'))
        if not os.path.exists(label_file):
            continue

        with open(label_file, 'r') as f:
            for idx, line in enumerate(f.readlines()):
                try:
                    class_num, xs, ys, ws, hs = line.strip().split(' ')
                    class_id, xc, yc, w, h = int(class_num), float(xs), float(ys), float(ws), float(hs)

                    # YOLO格式转COCO格式的bbox计算
                    xmin = (xc - w / 2) * width
                    ymin = (yc - h / 2) * height
                    bbox_w = int(width * w)
                    bbox_h = int(height * h)

                    annotations.append({
                        "category_id": class_id + 1,  # 类别ID偏移
                        "bbox": [xmin, ymin, bbox_w, bbox_h],
                        "area": bbox_w * bbox_h,
                        "image_id": index,
                        "id": index * 100 + idx,
                        "segmentation": [[
                            xmin, ymin,
                            xmin + bbox_w, ymin,
                            xmin + bbox_w, ymin + bbox_h,
                            xmin, ymin + bbox_h
                        ]],
                        "iscrowd": 0
                    })
                except ValueError:
                    print(f"Invalid label format in {label_file}, line {idx}")

    return images, annotations


def copy_image_files(
        src_images_path: str,
        dst_images_path: str,
        allowed_ext: tuple = ('.jpg', '.png', '.jpeg', '.bmp')
):
    """
    仅复制图片文件到目标目录
    :param allowed_ext: 允许复制的图片格式
    """
    os.makedirs(dst_images_path, exist_ok=True)
    copied_files = 0

    for img_file in os.listdir(src_images_path):
        if img_file.lower().endswith(allowed_ext):
            try:
                shutil.copy2(
                    src=os.path.join(src_images_path, img_file),
                    dst=os.path.join(dst_images_path, img_file)
                )
                copied_files += 1
            except Exception as e:
                print(f"Failed to copy {img_file}: {str(e)}")

    print(f"Copied {copied_files} images to {dst_images_path}")


def generate_coco_dataset(
        yolo_images_path: str,
        yolo_labels_path: str,
        output_dir: str,
        dataset_type: str,
        class_path: Optional[str] = None,
        copy_files: bool = True  # 新增参数控制是否复制文件
):
    """
    生成COCO格式数据集（增加class_path参数）
    :param class_path: 可选，指定classes.txt 路径
    """
    os.makedirs(os.path.join(output_dir, 'annotations'), exist_ok=True)

    # 初始化数据结构（传入class_path）
    coco_data = init_coco_structure(class_path)

    images, annotations = convert_yolo_to_coco(yolo_images_path, yolo_labels_path)
    coco_data['images'] = images
    coco_data['annotations'] = annotations

    json_path = os.path.join(output_dir, f'annotations/instances_{dataset_type}.json')
    with open(json_path, 'w') as f:
        json.dump(coco_data, f, indent=2)

    # 复制原始文件（图片+标签）
    if copy_files:
        copy_image_files(
            src_images_path=yolo_images_path,
            dst_images_path=os.path.join(output_dir, dataset_type),
        )

    print(f"COCO {dataset_type} set generated at: {json_path}")


def yolo_to_coco_converter(
        yolo_images_path: str,
        yolo_labels_path: str,
        yolo_class_path: str,
        coco_output_path: str,
        dataset_types: List[str] = ['train', 'val'],
        copy_original_files: bool = True  # 新增控制参数
):
    """
    最终转换接口
    :param yolo_class_path: classes.txt 文件路径
    """
    for dtype in dataset_types:
        img_path = os.path.join(yolo_images_path, dtype)
        label_path = os.path.join(yolo_labels_path, dtype)

        if not all(os.path.exists(p) for p in [img_path, label_path]):
            print(f"Skipping {dtype} set (invalid paths)")
            continue

        generate_coco_dataset(
            yolo_images_path=img_path,
            yolo_labels_path=label_path,
            output_dir=coco_output_path,
            dataset_type=dtype,
            class_path=yolo_class_path,  # 传递类别文件路径
            copy_files=copy_original_files

        )


# 使用示例
if __name__ == '__main__':
    yolo_to_coco_converter(
        yolo_images_path=r'E:\pycharm\DataTransformer\data\images4',
        yolo_labels_path=r'E:\pycharm\DataTransformer\data\labels4',
        yolo_class_path=r'E:\pycharm\DataTransformer\data\classes.txt',  # 新增
        coco_output_path=r'E:\pycharm\DataTransformer\data\coco',
        dataset_types=['train', 'val'],
        copy_original_files=True  # 启用文件复制
    )