import json
import os
from typing import Dict


def load_label_mapping(class_file_path: str) -> Dict[str, int]:
    """
    从class.txt 文件加载类别到ID的映射关系

    Args:
        class_file_path: class.txt 文件路径，每行一个类别名称

    Returns:
        字典格式的映射关系 {类别名: 类别ID}

    Raises:
        FileNotFoundError: 当文件不存在时抛出
    """
    label_to_id = {}
    with open(class_file_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            class_name = line.strip()
            if class_name:
                label_to_id[class_name] = idx
    return label_to_id


def convert_annotation(
        json_file: str,
        txt_file: str,
        label_mapping: Dict[str, int],
        default_id: int = -1
) -> None:
    """
    将JSON标注文件转换为YOLO格式的TXT文件

    Args:
        json_file: 输入的JSON标注文件路径
        txt_file: 输出的YOLO格式TXT文件路径
        label_mapping: 类别名称到ID的映射字典
        default_id: 未匹配类别的默认ID（默认为-1）

    Raises:
        ValueError: 当JSON缺少必要字段时抛出
        json.JSONDecodeError: JSON解析错误时抛出
    """
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in {json_file}") from e

        # 验证必要字段
    required_fields = ['imageWidth', 'imageHeight', 'shapes']
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field '{field}' in {json_file}")

    with open(txt_file, 'w', encoding='utf-8') as out_file:
        for shape in data['shapes']:
            # 边界框计算
            points = shape.get('points', [])
            if len(points) < 2:
                print(f"Warning: Ignoring invalid shape in {json_file} (points: {points})")
                continue

            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            # 归一化坐标计算
            width_ratio = (x_max - x_min) / data['imageWidth']
            height_ratio = (y_max - y_min) / data['imageHeight']
            center_x = (x_min + x_max) / 2 / data['imageWidth']
            center_y = (y_min + y_max) / 2 / data['imageHeight']

            # 类别ID获取
            category = shape.get('label', 'unknown')
            category_id = label_mapping.get(category, default_id)

            # YOLO格式写入
            out_file.write(
                f"{category_id} {center_x:.6f} {center_y:.6f} {width_ratio:.6f} {height_ratio:.6f}\n"
            )


def batch_convert(
        input_folder: str,
        output_folder: str,
        class_file: str,
        verbose: bool = True
) -> None:
    """
    批量转换文件夹内的所有JSON标注文件

    Args:
        input_folder: 包含JSON文件的输入文件夹
        output_folder: 输出YOLO格式TXT的文件夹
        class_file: 类别定义文件路径
        verbose: 是否打印转换进度（默认为True）
    """
    os.makedirs(output_folder, exist_ok=True)

    try:
        label_mapping = load_label_mapping(class_file)
        if verbose:
            print(f"Loaded {len(label_mapping)} classes from {class_file}")
    except Exception as e:
        print(f"Warning: Failed to load class file ({e}), using empty mapping")
        label_mapping = {}

    for filename in os.listdir(input_folder):
        if filename.endswith('.json'):
            json_path = os.path.join(input_folder, filename)
            txt_path = os.path.join(output_folder, filename.replace('.json', '.txt'))

            try:
                convert_annotation(json_path, txt_path, label_mapping)
                if verbose:
                    print(f"Converted {filename} -> {os.path.basename(txt_path)}")
            except Exception as e:
                print(f"Error converting {filename}: {str(e)}")


# 使用示例
if __name__ == "__main__":
    # 配置路径（实际使用时修改这些参数）
    input_dir = r'E:\pycharm\DataTransformer\data\jsons'
    output_dir = r'E:\pycharm\DataTransformer\data\labels2'
    class_file = r'E:\pycharm\DataTransformer\data\class.txt'

    # 执行批量转换
    batch_convert(
        input_folder=input_dir,
        output_folder=output_dir,
        class_file=class_file
    )