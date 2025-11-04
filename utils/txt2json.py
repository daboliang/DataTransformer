import cv2
import os
import json
from typing import List, Dict, Tuple, Optional
import numpy as np

def load_label_mapping(class_file_path: str) -> Dict[int, str]:
    """
    从class.txt 文件加载ID到类别的映射关系

    Args:
        class_file_path: class.txt 文件路径，每行一个类别名称

    Returns:
        字典格式的映射关系 {类别ID: 类别名}

    Raises:
        FileNotFoundError: 当文件不存在时抛出
    """
    id_to_name = {}
    with open(class_file_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            class_name = line.strip()
            if class_name:
                id_to_name[idx] = class_name
    return id_to_name


def convert_yolo_bbox(box: List[float], img_shape: Tuple[int, int]) -> List[int]:
    """
    将YOLO格式的归一化坐标转换为像素坐标

    Args:
        box: [x_center, y_center, width, height] 归一化坐标
        img_shape: (height, width) 图像尺寸

    Returns:
        [x1, y1, x2, y2] 像素坐标
    """
    x_center, y_center, w, h = box
    img_h, img_w = img_shape[:2]
    x1 = int((x_center - w / 2) * img_w)
    y1 = int((y_center - h / 2) * img_h)
    x2 = int((x_center + w / 2) * img_w)
    y2 = int((y_center + h / 2) * img_h)
    return [x1, y1, x2, y2]


def parse_yolo_txt(
        img_path: str,
        txt_path: str,
        id_to_name: Dict[int, str]
) -> Tuple[Optional[np.ndarray], List[List]]:
    """
    解析YOLO标签文件

    Args:
        img_path: 图像文件路径
        txt_path: 标签文件路径
        id_to_name: 类别ID到名称的映射

    Returns:
        (图像数据, 标注列表) 标注格式: [[x1,y1,x2,y2,label], ...]

    Raises:
        FileNotFoundError: 当图像或标签文件不存在时抛出
        ValueError: 当标签格式错误时抛出
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {img_path}")

    bboxes = []
    if not os.path.exists(txt_path):
        return img, bboxes  # 返回空列表表示没有目标

    with open(txt_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            parts = line.strip().split()
            if len(parts) != 5:
                print(f"Warning: 跳过第{line_num}行(无效格式): {line.strip()}")
                continue

            try:
                label_idx = int(parts[0])
                label = id_to_name.get(label_idx, f"unknown_{label_idx}")
                box = list(map(float, parts[1:5]))
                pixel_box = convert_yolo_bbox(box, img.shape)
                pixel_box.append(label)
                bboxes.append(pixel_box)
            except (ValueError, IndexError) as e:
                print(f"Error in {txt_path} line {line_num}: {str(e)}")
                continue

    return img, bboxes


def create_labelme_json(
        img_name: str,
        img_dir: str,
        txt_dir: str,
        output_dir: str,
        id_to_name: Dict[int, str],
        save_image_data: bool = False
) -> Optional[str]:
    """
    创建Labelme格式的JSON标注文件

    Args:
        img_name: 图像文件名 (如 "image.jpg")
        img_dir: 图像目录路径
        txt_dir: 标签目录路径
        output_dir: 输出目录路径
        id_to_name: 类别ID到名称的映射
        save_image_data: 是否在JSON中保存图像base64数据

    Returns:
        生成的JSON文件路径 (失败时返回None)
    """
    # 构建完整路径
    img_path = os.path.join(img_dir, img_name)
    base_name = os.path.splitext(img_name)[0]
    txt_path = os.path.join(txt_dir, f"{base_name}.txt")

    # 解析数据
    try:
        img, gt_boxes = parse_yolo_txt(img_path, txt_path, id_to_name)
    except Exception as e:
        print(f"处理 {img_name} 时出错: {str(e)}")
        return None

        # 构建基础字典
    label_dict = {
        "version": "5.5.0",
        "flags": {},
        "shapes": [],
        "imagePath": os.path.basename(img_path),
        "imageData": None,
        "imageHeight": img.shape[0],
        "imageWidth": img.shape[1]
    }

    # 可选: 保存图像数据
    if save_image_data:
        import base64
        _, img_encoded = cv2.imencode('.jpg', img)
        label_dict["imageData"] = base64.b64encode(img_encoded).decode('utf-8')

    # 添加标注形状
    for box in gt_boxes:
        shape = {
            "label": box[-1],
            "points": [[box[0], box[1]], [box[2], box[3]]],
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {}
        }
        label_dict["shapes"].append(shape)

    # 写入JSON文件
    output_path = os.path.join(output_dir, f"{base_name}.json")
    os.makedirs(output_dir, exist_ok=True)

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(label_dict, f, indent=2, ensure_ascii=False)
        return output_path
    except Exception as e:
        print(f"写入 {output_path} 失败: {str(e)}")
        return None


def batch_convert_yolo_to_labelme(
        img_dir: str,
        txt_dir: str,
        output_dir: str,
        class_file: str,
        save_image_data: bool = False,
        verbose: bool = True
) -> None:
    """
    批量转换YOLO格式到Labelme格式

    Args:
        img_dir: 图像目录路径
        txt_dir: 标签目录路径
        output_dir: 输出目录路径
        class_file: 类别定义文件路径
        save_image_data: 是否在JSON中保存图像数据
        verbose: 是否打印详细进度
    """
    # 加载类别映射
    try:
        id_to_name = load_label_mapping(class_file)
        if verbose:
            print(f"加载 {len(id_to_name)} 个类别从 {class_file}")
    except Exception as e:
        print(f"加载类别文件失败: {str(e)}")
        id_to_name = {}

    # 获取所有图像文件
    img_files = [f for f in os.listdir(img_dir)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if verbose:
        print(f"发现 {len(img_files)} 张待处理图像")

    # 批量处理
    success_count = 0
    for idx, img_name in enumerate(img_files, 1):
        if verbose and idx % 100 == 0:
            print(f"进度: {idx}/{len(img_files)}")

        result = create_labelme_json(
            img_name=img_name,
            img_dir=img_dir,
            txt_dir=txt_dir,
            output_dir=output_dir,
            id_to_name=id_to_name,
            save_image_data=save_image_data
        )

        if result is not None:
            success_count += 1
            if verbose:
                print(f"生成: {os.path.basename(result)}")

    if verbose:
        print(f"\n转换完成! 成功: {success_count}/{len(img_files)}")


# 使用示例
if __name__ == "__main__":
    # 配置路径
    IMG_DIR = r"E:\pycharm\DataTransformer\data\images"
    TXT_DIR = r"E:\pycharm\DataTransformer\data\labels"
    OUTPUT_DIR = r"E:\pycharm\DataTransformer\data\labelme_annotations"
    CLASS_FILE = r"E:\pycharm\DataTransformer\data\class.txt"

    # 执行转换
    batch_convert_yolo_to_labelme(
        img_dir=IMG_DIR,
        txt_dir=TXT_DIR,
        output_dir=OUTPUT_DIR,
        class_file=CLASS_FILE,
        save_image_data=False,
        verbose=True
    )