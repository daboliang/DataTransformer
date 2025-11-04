import os
import random
import shutil


def split_dataset(
        images_path: str,
        labels_path: str,
        out_imagesPath: str,
        out_labelsPath: str,
        split_rate: float = 0.8,  # 训练集比例（验证集自动计算为 1-split_rate）
        seed: int = None
) -> dict:
    """
    分割数据集为训练集和验证集

    参数:
        images_path: 原始图像路径
        labels_path: 原始标注路径
        out_imagesPath: 输出图像根路径
        out_labelsPath: 输出标注根路径
        split_rate: 训练集比例（0-1之间，默认0.8）
        seed: 随机种子（可选）

    返回:
        包含各集合数量的字典 {'train': x, 'val': y}
    """
    # 参数校验
    if not 0 < split_rate < 1:
        raise ValueError("split_rate 必须在0到1之间")
    val_rate = 1 - split_rate

    # 设置随机种子
    if seed is not None:
        random.seed(seed)

        # 创建输出目录
    train_img_dir = os.path.join(out_imagesPath, "train")
    val_img_dir = os.path.join(out_imagesPath, "val")
    train_label_dir = os.path.join(out_labelsPath, "train")
    val_label_dir = os.path.join(out_labelsPath, "val")

    for dir_path in [train_img_dir, val_img_dir, train_label_dir, val_label_dir]:
        os.makedirs(dir_path, exist_ok=True)
        # 清空目录
        for f in os.listdir(dir_path):
            os.remove(os.path.join(dir_path, f))

    # 获取所有标注文件
    label_files = [f for f in os.listdir(labels_path) if f.endswith('.txt')]
    total = len(label_files)
    num_train = int(total * split_rate)

    # 随机分割
    indices = list(range(total))
    train_indices = random.sample(indices, num_train)
    val_indices = [i for i in indices if i not in train_indices]

    # 文件复制函数
    def copy_files(indices, img_dest, label_dest):
        for idx in indices:
            base_name = os.path.splitext(label_files[idx])[0]

            # 查找对应图像（支持多种格式）
            for ext in ['.jpg', '.jpeg', '.png']:
                src_img = os.path.join(images_path, base_name + ext)
                if os.path.exists(src_img):
                    shutil.copy(src_img, os.path.join(img_dest, base_name + ext))
                    shutil.copy(
                        os.path.join(labels_path, base_name + '.txt'),
                        os.path.join(label_dest, base_name + '.txt')
                    )
                    break

                    # 执行复制

    copy_files(train_indices, train_img_dir, train_label_dir)
    copy_files(val_indices, val_img_dir, val_label_dir)

    return {
        'train': len(train_indices),
        'val': len(val_indices),
        'train_ratio': split_rate,
        'val_ratio': val_rate
    }


# 使用示例
if __name__ == "__main__":
    # 直接传递参数
    result = split_dataset(
        images_path="path/to/images",
        labels_path="path/to/labels",
        out_imagesPath="output/images",
        out_labelsPath="output/labels",
        split_rate=0.8,  # 80%训练集
        seed=42
    )
    print(f"分割结果: {result}")