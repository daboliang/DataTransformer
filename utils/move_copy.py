import os
import shutil
from pathlib import Path
from typing import List, Union


def move_copy(
        source_folder: str,
        target_folder: str,
        operation_mode: int = 0,
        extensions: Union[List[str], None] = None
) -> bool:
    """
    文件操作函数（复制/移动 + 后缀筛选）

    参数:
        source_folder (str): 源文件夹路径
        target_folder (str): 目标文件夹路径
        operation_mode (int): 操作模式 (0=复制，1=移动)，默认为0
        extensions (List[str]|None): 要操作的文件后缀列表，None表示所有文件

    返回:
        bool: 操作是否成功完成

    功能:
        1. 支持复制或移动两种操作模式
        2. 可选按文件后缀筛选
        3. 自动创建目标目录
        4. 完整错误处理和操作统计
    """
    try:
        # 转换为Path对象
        src = Path(source_folder)
        dst = Path(target_folder)

        # 验证源文件夹
        if not src.exists() or not src.is_dir():
            print(f"错误: 源文件夹 {source_folder} 不存在或不是文件夹")
            return False

            # 验证操作模式
        if operation_mode not in (0, 1):
            print("错误: operation_mode 参数必须是 0(复制) 或 1(移动)")
            return False

            # 创建目标文件夹(如果不存在)
        dst.mkdir(parents=True, exist_ok=True)

        # 处理extensions参数
        if extensions is not None:
            # 统一转换为小写并确保以点开头
            extensions = [ext.lower() if ext.startswith('.') else f'.{ext.lower()}'
                          for ext in extensions]
            print(f"筛选条件: 只处理 {extensions} 类型的文件")

        # 操作类型描述
        operation = "复制" if operation_mode == 0 else "移动"
        print(f"\n开始{operation}操作: {source_folder} -> {target_folder}")

        # 文件计数器
        processed_count = 0
        skipped_count = 0

        # 遍历源文件夹
        for item in src.iterdir():
            if item.is_file():
                # 检查后缀是否符合要求
                if extensions is None or item.suffix.lower() in extensions:
                    try:
                        # 根据模式选择操作
                        if operation_mode == 0:
                            shutil.copy2(str(item), str(dst / item.name))
                        else:
                            shutil.move(str(item), str(dst / item.name))

                        print(f"已{operation}: {item.name}")
                        processed_count += 1
                    except Exception as e:
                        print(f"{operation} {item.name}  失败: {e}")
                else:
                    skipped_count += 1

                    # 操作结果统计
        print(f"\n{operation}操作完成！共处理 {processed_count + skipped_count} 个文件")
        print(f"成功{operation}: {processed_count} 个")
        print(f"跳过: {skipped_count} 个(不符合后缀条件)")
        return True

    except Exception as e:
        print(f"\n操作失败: {e}")
        return False