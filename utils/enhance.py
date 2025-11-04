import cv2
import os
from pathlib import Path
import numpy as np


def enhance_high_frequency(input_dir, output_dir, kernel_size=(5, 5), alpha=1.5, beta=-0.5):
    """
    通过拉普拉斯算子增强图像高频特征（适合纹理/边缘敏感任务）

    参数:
        input_dir: 输入文件夹路径（支持.jpg/.png）
        output_dir: 增强后保存路径（自动创建）
        kernel_size: 高斯模糊核大小
        alpha: 锐化强度（建议1.0-2.0）
        beta: 原始图像保留权重（负值强化边缘）
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    for img_path in input_dir.glob('*'):
        if img_path.suffix.lower()  in {'.jpg', '.jpeg', '.png'}:
            # 读取（支持中文）
            img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)

            # 高斯模糊后与原图做差提取高频
            blurred = cv2.GaussianBlur(img, kernel_size, 0)
            high_freq = cv2.addWeighted(img, alpha, blurred, beta, 0)

            # 保存（支持中文）
            output_path = output_dir / f"enhanced_{img_path.name}"
            cv2.imencode(output_path.suffix,  high_freq)[1].tofile(str(output_path))

    print(f"高频增强完成！结果保存在 {output_dir}")

def normalize_illumination(input_dir, output_dir, clip_limit=2.0, grid_size=(8, 8), gamma=0.7):
    """
    解决光照不均问题（适合低光照/过曝图像）

    参数:
        clip_limit: CLAHE对比度阈值（建议1-3）
        grid_size: 直方图均衡化分块数
        gamma: 校正强度（<1提亮，>1压暗）
    """

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)

    for img_path in input_dir.glob('*'):
        if img_path.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
            # 读取（支持中文）
            img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)

            # 转换到LAB空间处理亮度通道
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l_norm = clahe.apply(l)

            # Gamma校正
            l_gamma = cv2.pow(l_norm / 255.0, gamma) * 255.0
            enhanced_lab = cv2.merge((l_gamma.astype('uint8'), a, b))

            # 转回BGR保存
            result = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

            # 保存（支持中文）
            output_path = output_dir / f"norm_{img_path.name}"
            cv2.imencode(output_path.suffix, result)[1].tofile(str(output_path))

    print(f"光照归一化完成！结果保存在 {output_dir}")


def multi_scale_fusion(input_dir, output_dir, levels=3, blend_weights=[0.5, 0.3, 0.2]):
    """
    融合不同尺度的特征（适合多尺度目标检测）

    参数:
        levels: 金字塔层数（建议3-5）
        blend_weights: 各层融合权重（需sum=1.0）
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    for img_path in input_dir.glob('*'):
        if img_path.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
            # 读取（支持中文）
            img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
            h, w = img.shape[:2]

            # 生成高斯金字塔
            pyramid = [img]
            for _ in range(levels - 1):
                img = cv2.pyrDown(img)
                pyramid.append(img)

                # 上采样并加权融合
            fused = pyramid[-1]
            for i in range(levels - 2, -1, -1):
                fused = cv2.pyrUp(fused, dstsize=(pyramid[i].shape[1], pyramid[i].shape[0]))
                fused = cv2.addWeighted(pyramid[i], blend_weights[i], fused, 1 - blend_weights[i], 0)


            # 保存（支持中文）
            output_path = output_dir / f"fused_{img_path.name}"
            cv2.imencode(output_path.suffix, fused)[1].tofile(str(output_path))

    print(f"多尺度融合完成！结果保存在 {output_dir}")