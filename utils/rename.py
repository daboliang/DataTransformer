import os
import json
import shutil
from pathlib import Path


def rename_files(images_path=None, labels_path=None, jsons_path=None, start_index=1, fixed_name=""):
    """
    通用文件重命名函数，支持多类型文件同步重命名和JSON内容更新

    参数:
        images_path: 图像文件夹路径（可选）
        labels_path: 标签文件夹路径（可选）
        jsons_path: JSON文件夹路径（可选）
        start_index: 起始序号（整型）
        fixed_name: 附加的固定名称信息
    """
    # 确定需要处理的文件夹列表（过滤掉None值）
    target_dirs = [d for d in [images_path, labels_path, jsons_path] if d is not None and os.path.exists(d)]
    if not target_dirs:
        raise ValueError("至少需要提供一个有效路径！")

    # 获取各文件夹中的文件列表（按名称排序，且只包含文件）
    file_groups = {}
    for dir_path in target_dirs:
        files = sorted([
            f for f in os.listdir(dir_path)
            if not f.startswith('.') and os.path.isfile(os.path.join(dir_path, f))  # 排除隐藏文件和子文件夹
        ])
        file_groups[dir_path] = files

        # 检查文件数量是否一致（如果多个路径都存在）
    if len(target_dirs) > 1:
        file_counts = [len(files) for files in file_groups.values()]
        if len(set(file_counts)) > 1:
            print(f"警告：文件数量不一致！{dict(zip(target_dirs, file_counts))}")

    # 创建临时备份文件夹（用于JSON文件）
    backup_dirs = {}
    for dir_path in target_dirs:
        backup_dir = os.path.join(dir_path, "rename_backup")
        os.makedirs(backup_dir, exist_ok=True)
        backup_dirs[dir_path] = backup_dir
        for f in file_groups[dir_path]:
            shutil.copy2(os.path.join(dir_path, f), backup_dir)

    # 获取基准文件数量（以第一个文件夹为准）
    base_files = file_groups[target_dirs[0]]
    total_files = len(base_files)

    # 开始重命名处理
    for i in range(total_files):
        current_index = start_index + i
        new_basename = f"{current_index}_{fixed_name}" if fixed_name else str(current_index)

        # 处理每个文件夹中的对应文件
        for dir_path in target_dirs:
            old_filename = file_groups[dir_path][i]
            old_path = os.path.join(dir_path, old_filename)

            # 获取文件扩展名
            file_ext = os.path.splitext(old_filename)[1]
            new_filename = f"{new_basename}{file_ext}"
            new_path = os.path.join(dir_path, new_filename)

            # 执行重命名
            os.rename(old_path, new_path)
            print(f"重命名: {old_filename} -> {new_filename}")

            # 如果是JSON文件，更新内部引用的图像路径
            if dir_path == jsons_path and file_ext.lower() == '.json':
                with open(new_path, 'r+', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                        if "imagePath" in data:
                            # 更新imagePath字段（保持原扩展名或使用新命名）
                            img_ext = os.path.splitext(data["imagePath"])[1] or file_ext
                            data["imagePath"] = f"{new_basename}{img_ext}"
                            f.seek(0)
                            json.dump(data, f, indent=4, ensure_ascii=False)
                            f.truncate()
                    except json.JSONDecodeError:
                        print(f"警告：无法解析JSON文件 {new_path}")

    print("\n操作完成！原始文件备份保存在：")
    for dir_path, backup_dir in backup_dirs.items():
        print(f"- {backup_dir}")


# 使用示例（适配您的新参数结构）
if __name__ == "__main__":
    class MockMain:
        def __init__(self):
            self.imagesPath = r"E:\pycharm\ultralytics-main\ultralytics\hands_dataset\1"
            self.labelsPath = r"E:\pycharm\ultralytics-main\ultralytics\hands_dataset\2"
            self.jsonsPath = r"E:\pycharm\ultralytics-main\ultralytics\hands_dataset\3"
            self.FirstName = 52  # 起始编号
            self.SecondName = "newname"  # 固定名称


    main = MockMain()
    rename_files(
        main.imagesPath,
        main.labelsPath,
        main.jsonsPath,
        main.FirstName,
        main.SecondName
    )