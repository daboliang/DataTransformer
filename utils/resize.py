import cv2
import numpy as np
from pathlib import Path


def yolov8_style_resize(images_dir, labels_dir, target_size=1024, color=(114, 114, 114)):
    """
    YOLOv8风格的图像预处理：
    1. 原图按长边比例缩放至target_size
    2. 短边用指定颜色填充
    3. 若原图长边超过target_size，则先等比缩放到target_size再填充
    """
    # 创建输出目录
    output_img_dir = Path(images_dir).parent / "resized_images"
    output_label_dir = Path(labels_dir).parent / "resized_labels"
    output_img_dir.mkdir(exist_ok=True)
    output_label_dir.mkdir(exist_ok=True)

    for img_path in Path(images_dir).glob("*"):
        if img_path.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.bmp'):
            continue

            # 读取原始图像和标签
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        label_path = Path(labels_dir) / f"{img_path.stem}.txt"

        # YOLOv8预处理逻辑
        scale = min(target_size / h, target_size / w)
        if scale < 1:  # 仅当原图大于目标尺寸时才缩小
            new_h, new_w = int(h * scale), int(w * scale)
            img = cv2.resize(img, (new_w, new_h))
        else:
            new_h, new_w = h, w

        # 创建填充后的画布
        canvas = np.full((target_size, target_size, 3), color, dtype=np.uint8)
        # 计算填充位置（居中）
        top = (target_size - new_h) // 2
        left = (target_size - new_w) // 2
        canvas[top:top + new_h, left:left + new_w] = img

        # 保存处理后的图像
        cv2.imwrite(str(output_img_dir / img_path.name), canvas)

        # 调整YOLO标签坐标
        if label_path.exists():
            with open(label_path, 'r') as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                cls, x, y, bw, bh = map(float, line.strip().split())
                # 坐标转换公式
                new_x = (x * new_w + left) / target_size
                new_y = (y * new_h + top) / target_size
                new_bw = bw * new_w / target_size
                new_bh = bh * new_h / target_size
                new_lines.append(f"{int(cls)}  {new_x:.6f} {new_y:.6f} {new_bw:.6f} {new_bh:.6f}\n")

            # 保存调整后的标签
            with open(output_label_dir / label_path.name, 'w') as f:
                f.writelines(new_lines)

        print(f"Processed: {img_path.name}")
