import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
import cv2

def convert_yolo_to_voc(Yolo_imagesPath,Yolo_labelsPath,Yolo_classPath,VOC_RootPath):
    """核心转换函数（严格适配用户指定的YOLO结构）"""
    print("\n🔍 今天是2025年11月4日 星期二 09:59")
    print("🔄 开始转换YOLO→VOC格式（适配新版结构）...")

    # 加载类别文件
    try:
        with open(Yolo_classPath, 'r', encoding='utf-8') as f:
            classes = [line.strip() for line in f if line.strip()]
        print(f"📋 加载 {len(classes)} 个类别: {classes}")
    except FileNotFoundError:
        raise Exception(f"❌ 类别文件不存在: {Yolo_classPath}")

    # 创建VOC标准目录
    os.makedirs(os.path.join(VOC_RootPath, 'JPEGImages'), exist_ok=True)
    os.makedirs(os.path.join(VOC_RootPath, 'Annotations'), exist_ok=True)
    os.makedirs(os.path.join(VOC_RootPath, 'ImageSets', 'Main'), exist_ok=True)

    # 处理每个划分集
    for split in ['train', 'val']:
        print(f"\n📂 处理 {split} 数据集...")
        img_dir = os.path.join(Yolo_imagesPath, split)
        label_dir = os.path.join(Yolo_labelsPath, split)

        if not os.path.exists(img_dir):
            print(f"⚠️ 跳过 {split} - 图像目录不存在: {img_dir}")
            continue

            # 获取图像列表（兼容多种格式）
        img_files = [f for f in os.listdir(img_dir)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

        # 处理每个样本
        split_samples = []
        for img_file in img_files:
            base_name = os.path.splitext(img_file)[0]
            img_path = os.path.join(img_dir, img_file)
            label_path = os.path.join(label_dir, f"{base_name}.txt")

            # 复制图像到JPEGImages
            cv2.imwrite(os.path.join(VOC_RootPath, 'JPEGImages', img_file),
                        cv2.imread(img_path))

            # 生成XML标注
            if os.path.exists(label_path):
                annotation = ET.Element('annotation')
                # 添加基础信息（简化版）
                ET.SubElement(annotation, 'filename').text = img_file
                ET.SubElement(annotation, 'folder').text = 'JPEGImages'

                # 添加尺寸信息
                img = cv2.imread(img_path)
                h, w = img.shape[:2]
                size = ET.SubElement(annotation, 'size')
                ET.SubElement(size, 'width').text = str(w)
                ET.SubElement(size, 'height').text = str(h)
                ET.SubElement(size, 'depth').text = '3'

                # 处理YOLO标签
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            obj = ET.SubElement(annotation, 'object')
                            ET.SubElement(obj, 'name').text = classes[int(parts[0])]
                            ET.SubElement(obj, 'difficult').text = '0'

                            # 坐标转换
                            x_center, y_center, bw, bh = map(float, parts[1:])
                            xmin = int((x_center - bw / 2) * w)
                            ymin = int((y_center - bh / 2) * h)
                            xmax = int((x_center + bw / 2) * w)
                            ymax = int((y_center + bh / 2) * h)

                            bndbox = ET.SubElement(obj, 'bndbox')
                            ET.SubElement(bndbox, 'xmin').text = str(max(0, xmin))
                            ET.SubElement(bndbox, 'ymin').text = str(max(0, ymin))
                            ET.SubElement(bndbox, 'xmax').text = str(min(w - 1, xmax))
                            ET.SubElement(bndbox, 'ymax').text = str(min(h - 1, ymax))

                # 保存XML
                xml_str = minidom.parseString(ET.tostring(annotation)).toprettyxml(indent="   ")
                with open(os.path.join(VOC_RootPath, 'Annotations', f"{base_name}.xml"), 'w') as f:
                    f.write(xml_str)

            split_samples.append(base_name)

            # 生成划分文件
        with open(os.path.join(VOC_RootPath, 'ImageSets', 'Main', f'{split}.txt'), 'w') as f:
            f.write('\n'.join(split_samples))

        print(f"✅ {split} 完成: {len(split_samples)} 个样本")

    print("\n🎉 转换结果验证:")
    print(f"JPEGImages/: {len(os.listdir(os.path.join(VOC_RootPath, 'JPEGImages')))} 图像")
    print(f"Annotations/: {len(os.listdir(os.path.join(VOC_RootPath, 'Annotations')))} 标注")
    print(f"ImageSets/Main/: {os.listdir(os.path.join(VOC_RootPath, 'ImageSets', 'Main'))}")


if __name__ == "__main__":
    # -----------------------------
    # 全局路径配置（2025年11月4日 09:49 更新）
    # -----------------------------
    Yolo_imagesPath = r'E:\pycharm\DataTransformer\data\images4'  # YOLO图像根目录
    Yolo_labelsPath = r'E:\pycharm\DataTransformer\data\labels4'  # YOLO标签根目录
    Yolo_classPath = r'E:\pycharm\DataTransformer\data\classes.txt'  # 类别文件
    VOC_RootPath = r'E:\pycharm\DataTransformer\data\VOC'  # VOC输出目录
    convert_yolo_to_voc(Yolo_imagesPath,Yolo_labelsPath,Yolo_classPath,VOC_RootPath)