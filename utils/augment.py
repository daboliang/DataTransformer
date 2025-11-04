import cv2
import os
import numpy as np
from pathlib import Path
import shutil
from typing import Union


def read_image_cn(path: Path) -> np.ndarray:
    """支持中文路径的图像读取"""
    raw_data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(raw_data, cv2.IMREAD_COLOR)


def write_image_cn(img: np.ndarray, path: Path):
    """支持中文路径的图像保存"""
    suffix = path.suffix.lower()
    if suffix == '.jpg':
        suffix = '.jpeg'  # OpenCV要求
    ret, buf = cv2.imencode(suffix, img)
    if ret:
        buf.tofile(str(path))


def get_enclosing_bbox(points: np.ndarray, img_w: int, img_h: int) -> Union[tuple, None]:
    """
    计算变换后的有效边界框
    :param points: 变换后的角点坐标 (N,2)
    :return: (x_min, y_min, x_max, y_max) 或 None(无效时)
    """
    x_min = max(0, min(points[:, 0]))
    y_min = max(0, min(points[:, 1]))
    x_max = min(img_w, max(points[:, 0]))
    y_max = min(img_h, max(points[:, 1]))
    return (x_min, y_min, x_max, y_max) if x_max > x_min and y_max > y_min else None


def geometric_augmentation(
        input_dir: Path,
        label_dir: Path,
        output_img_dir: Path,
        output_label_dir: Path,
        rotate_range: float = 30,
        flip_prob: float = 0.5,
        translate_ratio: float = 0.1,
        debug: bool = False
) -> None:
    """
    改进版几何增强函数（解决标签对齐问题）

    参数说明：
    - rotate_range: 旋转角度范围（-30~30度）
    - flip_prob: 水平翻转概率（0-1）
    - translate_ratio: 平移比例（相对于图像尺寸）
    - debug: 是否显示调试可视化
    """
    # 参数验证
    assert 0 <= flip_prob <= 1, "flip_prob必须在[0,1]范围内"
    assert translate_ratio >= 0, "translate_ratio必须非负"

    # 创建输出目录
    input_dir = Path(input_dir)
    label_dir = Path(label_dir)
    output_img_dir = Path(output_img_dir)
    output_label_dir = Path(output_label_dir)
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    # 获取所有图像文件
    img_paths = [p for p in input_dir.glob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}]

    for img_path in img_paths:
        # 读取图像
        img = read_image_cn(img_path)
        if img is None:
            print(f"⚠️ 图像读取失败: {img_path}")
            continue

        h, w = img.shape[:2]
        label_path = label_dir / f"{img_path.stem}.txt"

        # 生成随机变换参数
        angle = np.random.uniform(-rotate_range, rotate_range)
        flip = np.random.rand() < flip_prob
        tx = np.random.uniform(-translate_ratio, translate_ratio) * w
        ty = np.random.uniform(-translate_ratio, translate_ratio) * h

        # 构建变换矩阵
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        M[:, 2] += [tx, ty]  # 添加平移

        # 执行图像变换
        img_aug = cv2.warpAffine(
            img, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101
        )
        if flip:
            img_aug = cv2.flip(img_aug, 1)

        # 处理标签文件
        new_lines = []
        if label_path.exists():
            try:
                with open(str(label_path), 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue

                        cls, x, y, bw, bh = map(float, parts)
                        # 转换为绝对坐标
                        x_abs = x * w
                        y_abs = y * h
                        bw_abs = bw * w
                        bh_abs = bh * h

                        # 生成边界框角点（顺时针顺序）
                        corners = np.array([
                            [x_abs - bw_abs / 2, y_abs - bh_abs / 2],  # 左上
                            [x_abs + bw_abs / 2, y_abs - bh_abs / 2],  # 右上
                            [x_abs + bw_abs / 2, y_abs + bh_abs / 2],  # 右下
                            [x_abs - bw_abs / 2, y_abs + bh_abs / 2]  # 左下
                        ])

                        # 应用变换矩阵
                        ones = np.ones(shape=(4, 1))
                        points_ones = np.hstack([corners, ones])
                        transformed = M.dot(points_ones.T).T

                        # 处理翻转
                        if flip:
                            transformed[:, 0] = w - transformed[:, 0]

                        # 计算新边界框
                        bbox = get_enclosing_bbox(transformed, w, h)
                        if bbox is None:
                            continue  # 跳过无效标注

                        x_new, y_new, x2_new, y2_new = bbox
                        # 转换回YOLO格式
                        bw_new = (x2_new - x_new) / w
                        bh_new = (y2_new - y_new) / h
                        x_center = (x_new + x2_new) / 2 / w
                        y_center = (y_new + y2_new) / 2 / h

                        new_lines.append(f"{int(cls)}  {x_center:.6f} {y_center:.6f} {bw_new:.6f} {bh_new:.6f}\n")

            except Exception as e:
                print(f"标签处理错误 {label_path}: {str(e)}")

        # 调试可视化
        if debug and new_lines:
            debug_img = img_aug.copy()
            for line in new_lines:
                cls, x, y, bw, bh = map(float, line.strip().split())
                x1 = int((x - bw / 2) * w)
                y1 = int((y - bh / 2) * h)
                x2 = int((x + bw / 2) * w)
                y2 = int((y + bh / 2) * h)
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(debug_img, str(int(cls)), (x1 + 5, y1 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow("Augmentation  Debug", debug_img)
            cv2.waitKey(500 if debug else 1)

        # 保存结果
        if new_lines:  # 只保存有有效标注的结果
            output_img_path = output_img_dir / f"aug_{img_path.stem}{img_path.suffix}"
            output_label_path = output_label_dir / f"aug_{img_path.stem}.txt"

            write_image_cn(img_aug, output_img_path)
            with open(str(output_label_path), 'w', encoding='utf-8') as f:
                f.writelines(new_lines)


    print(f"✅ 几何增广完成！共处理 {len(img_paths)} 张图像")


def color_augmentation(input_dir: Path, label_dir: Path,
                       output_img_dir: Path, output_label_dir: Path,
                       brightness_range=0.2, contrast_range=0.2, noise_std=10):
    """
    仅修改图像色彩（标签文件直接复制）- 中文兼容版

    参数:
        brightness_range: 亮度调整范围（±比例）
        contrast_range: 对比度调整范围（±比例）
        noise_std: 高斯噪声标准差
    """
    input_dir = Path(input_dir)
    label_dir = Path(label_dir)
    output_img_dir = Path(output_img_dir)
    output_label_dir = Path(output_label_dir)
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    for img_path in input_dir.glob('*'):
        if img_path.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
            continue

            # 读取图像（中文兼容）
        img = read_image_cn(img_path)
        if img is None:
            print(f"⚠️ 图像读取失败: {img_path}")
            continue

            # 随机色彩变换
        alpha = 1 + np.random.uniform(-contrast_range, contrast_range)
        beta = np.random.uniform(-brightness_range, brightness_range) * 255
        img_aug = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        # 添加高斯噪声
        noise = np.random.normal(0, noise_std, img_aug.shape).astype('int16')
        img_aug = np.clip(img_aug.astype('int16') + noise, 0, 255).astype('uint8')

        # 保存结果（中文兼容）
        output_img_path = output_img_dir / f"aug_color_{img_path.name}"
        output_label_path = output_label_dir / f"aug_color_{img_path.stem}.txt"
        write_image_cn(img_aug, output_img_path)

        # 标签文件复制（中文兼容）
        try:
            label_path = label_dir / f"{img_path.stem}.txt"
            shutil.copy2(str(label_path), str(output_label_path))
        except Exception as e:
            print(f"标签复制错误 {label_path}: {str(e)}")

    print(f"色彩增广完成！")


def advanced_mixing(input_dir: Path, label_dir: Path,
                    output_img_dir: Path, output_label_dir: Path,
                    mosaic_size=640):
    """
    混合增广：随机选择4张图像生成Mosaic（需确保label_dir为YOLO格式）- 中文兼容版

    参数:
        mosaic_size: 输出图像尺寸（正方形）
    """
    input_dir = Path(input_dir)
    label_dir = Path(label_dir)
    output_img_dir = Path(output_img_dir)
    output_label_dir = Path(output_label_dir)
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    img_files = [f for f in input_dir.glob('*') if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
    for i in range(len(img_files) // 4):  # 每4张图生成1个Mosaic
        selected = np.random.choice(img_files, 4, replace=False)
        mosaic = np.zeros((mosaic_size, mosaic_size, 3), dtype=np.uint8)
        all_labels = []

        for j, img_path in enumerate(selected):
            # 读取图像（中文兼容）
            img = read_image_cn(img_path)
            if img is None:
                print(f"⚠️ 图像读取失败: {img_path}")
                continue

                # 随机位置拼接（2x2网格）
            x1, y1 = (j % 2) * mosaic_size // 2, (j // 2) * mosaic_size // 2
            x2, y2 = x1 + mosaic_size // 2, y1 + mosaic_size // 2
            img_resized = cv2.resize(img, (mosaic_size // 2, mosaic_size // 2))
            mosaic[y1:y2, x1:x2] = img_resized

            # 转换标签坐标（中文兼容）
            try:
                label_path = label_dir / f"{img_path.stem}.txt"
                with open(str(label_path), 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                for line in lines:
                    cls, x, y, bw, bh = map(float, line.strip().split())
                    x = x * mosaic_size // 2 + x1
                    y = y * mosaic_size // 2 + y1
                    bw = bw * mosaic_size // 2
                    bh = bh * mosaic_size // 2
                    all_labels.append(f"{int(cls)}   {x / mosaic_size:.6f} {y / mosaic_size:.6f} {bw / mosaic_size:.6f} {bh / mosaic_size:.6f}\n")
            except Exception as e:
                print(f"标签处理错误 {label_path}: {str(e)}")

        # 保存结果（中文兼容）
        output_img_path = output_img_dir / f"aug_mosaic_{i}.jpg"
        output_label_path = output_label_dir / f"aug_mosaic_{i}.txt"
        write_image_cn(mosaic, output_img_path)
        with open(str(output_label_path), 'w', encoding='utf-8') as f:
            f.writelines(all_labels)

    print(f"Mosaic增广完成！生成 {len(img_files) // 4} 个样本")