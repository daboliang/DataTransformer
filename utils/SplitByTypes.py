import os
import shutil
from pathlib import Path


def SplitByTypes(folder_path):
    """
    将文件夹中的文件按类型分类到不同子文件夹

    参数:
        folder_path (str): 要整理的文件夹路径

    功能:
        1. 扫描文件夹中的所有文件
        2. 根据文件扩展名创建对应的子文件夹
        3. 将文件移动到相应的子文件夹中
    """
    # 将输入路径转换为Path对象
    folder = Path(folder_path)

    # 确保路径存在且是文件夹
    if not folder.exists() or not folder.is_dir():
        print(f"错误: {folder_path} 不是一个有效的文件夹路径")
        return

    # 定义常见文件类型的分类文件夹名称
    type_folders = {
        # 图片类型
        '.jpg': 'images',
        '.jpeg': 'images',
        '.png': 'images',
        '.gif': 'images',
        '.bmp': 'images',
        '.webp': 'images',
        '.svg': 'images',
        '.tiff': 'images',

        # 文本类型
        '.txt': 'texts',
        '.md': 'texts',
        '.log': 'texts',

        # 数据文件
        '.json': 'json_files',
        '.csv': 'csv_files',
        '.xml': 'xml_files',

        # 文档
        '.pdf': 'pdfs',
        '.doc': 'docs',
        '.docx': 'docs',
        '.xls': 'spreadsheets',
        '.xlsx': 'spreadsheets',
        '.ppt': 'presentations',
        '.pptx': 'presentations',

        # 压缩文件
        '.zip': 'archives',
        '.rar': 'archives',
        '.7z': 'archives',
        '.tar': 'archives',
        '.gz': 'archives',

        # 可执行文件
        '.exe': 'executables',
        '.msi': 'executables',
        '.bat': 'executables',
        '.sh': 'executables',

        # 其他
        '.html': 'web_files',
        '.htm': 'web_files',
        '.css': 'web_files',
        '.js': 'web_files',
    }

    # 遍历文件夹中的所有文件
    for item in folder.iterdir():
        if item.is_file():
            # 获取文件扩展名(小写形式)
            ext = item.suffix.lower()

            # 确定目标文件夹名称
            target_folder_name = type_folders.get(ext, 'others')
            target_folder = folder / target_folder_name

            # 如果目标文件夹不存在则创建
            target_folder.mkdir(exist_ok=True)

            # 移动文件
            try:
                shutil.move(str(item), str(target_folder / item.name))
                print(f"已移动: {item.name}  -> {target_folder_name}/")
            except Exception as e:
                print(f"移动 {item.name}  失败: {e}")

    print("文件分类完成!")