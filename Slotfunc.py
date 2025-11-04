from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMessageBox, QFileDialog
from PyQt5.QtGui import QColor, QPainter, QPixmap
from collections import defaultdict

import os
from pathlib import Path



class SlotFunctions:
    def __init__(self, main_window):
        self.main = main_window  # 保留主窗口引用
        self.mapping = None

    # ----------- 文件操作槽函数 -----------
    def load_Path(self, line_edit):
        path = QFileDialog.getExistingDirectory(
            self.main, "选择路径", "", QFileDialog.ShowDirsOnly)
        if path:
            line_edit.setText(path)
            # 通过lineEdit.objectName() 判断不同类型路径的更新
            # 自动填充子路径
            if "Yolo_RootPath" in line_edit.objectName():
                self.main.ui.Yolo_imagesPath_lineEdit.setText(path + "/images")
                self.main.ui.Yolo_labelsPath_lineEdit.setText(path + "/labels")
                self.main.ui.Yolo_classPath_lineEdit.setText(path + "/classes.txt")
            elif "COCO_RootPath" in line_edit.objectName():
                self.main.ui.COCO_trainPath_lineEdit.setText(path + "/train")
                self.main.ui.COCO_valPath_lineEdit.setText(path + "/val")
                self.main.ui.COCO_annotationsPath_lineEdit.setText(path + "/annotations")
            elif "VOC_RootPath" in line_edit.objectName():
                self.main.ui.VOC_AnnotationsPath_lineEdit.setText(path + "/Annotations")
                self.main.ui.VOC_JPEGImagesPath_lineEdit.setText(path + "/JPEGImages")
                self.main.ui.VOC_ImageSetsPath_lineEdit.setText(path + "/ImageSets")

    def load_FilePath(self, line_edit):
        file_filter = "文本文件 (*.txt)"

        # 弹出文件选择对话框
        file_path, _ = QFileDialog.getOpenFileName(
            self.main,
            "选择文件",
            "",  # 默认目录
            file_filter,
            options=QFileDialog.Options()
        )

        # 如果选择了文件（支持多选）
        if file_path:
            line_edit.setText(file_path)

    def set_txt(self, lineEdit):
        # 统一通过lineEdit更新对应变量
        setattr(self.main, lineEdit.objectName().replace("_lineEdit", ""), lineEdit.text())

        # 2. 如果是OriginFolder_lineEdit触发，则统计后缀
        if lineEdit.objectName() == "OriginFloder_lineEdit":
            folder_path = lineEdit.text()

            # 清空原有数据
            self.main.Suffix = []
            self.main.ui.comboBox_Suffix.clear()

            # 检查文件夹是否存在
            if not os.path.isdir(folder_path):
                self.main.ui.comboBox_Suffix.addItem(" 文件夹不存在")
                return

            # 统计后缀类型
            suffix_counter = defaultdict(int)
            try:
                for entry in Path(folder_path).iterdir():
                    if entry.is_file():
                        suffix = entry.suffix.lower()
                        if suffix:  # 忽略无后缀文件
                            suffix_counter[suffix] += 1
            except Exception as e:
                self.main.ui.comboBox_Suffix.addItem(f" 读取错误: {str(e)}")
                return

                # 按出现频率排序(降序)
            sorted_suffixes = sorted(suffix_counter.items(),
                                     key=lambda x: x[1],
                                     reverse=True)

            # 保存并显示结果
            self.main.Suffix = [suffix[0] for suffix in sorted_suffixes]
            self.main.Suffix.append('None')

            # 添加到comboBox，显示格式为".txt (15个文件)"
            for suffix in self.main.Suffix:
                self.main.ui.comboBox_Suffix.addItem(suffix)
            self.main.ui.comboBox_Suffix.setCurrentIndex(len(self.main.Suffix)-1)

            # 如果没有找到有效后缀
            if not self.main.Suffix:
                return


    def set_float(self, doubleSpinBox):
        # 统一通过lineEdit更新对应变量
        setattr(self.main, doubleSpinBox.objectName().replace("doubleSpinBox_", ""), doubleSpinBox.value())

    def set_int(self, spinBox):
        # 统一通过lineEdit更新对应变量
        setattr(self.main, spinBox.objectName().replace("spinBox_", ""), spinBox.value())


    def Set_mode(self):
        if self.main.ui.YCV_comboBox1.currentText() == "Yolo" and self.main.ui.YCV_comboBox2.currentText() == "COCO":
            self.main.mode = 1
        elif self.main.ui.YCV_comboBox1.currentText() == "Yolo" and self.main.ui.YCV_comboBox2.currentText() == "VOC":
            self.main.mode = 2
        elif self.main.ui.YCV_comboBox1.currentText() == "COCO" and self.main.ui.YCV_comboBox2.currentText() == "Yolo":
            self.main.mode = 3
        elif self.main.ui.YCV_comboBox1.currentText() == "COCO" and self.main.ui.YCV_comboBox2.currentText() == "VOC":
            self.main.mode = 4
        elif self.main.ui.YCV_comboBox1.currentText() == "VOC" and self.main.ui.YCV_comboBox2.currentText() == "COCO":
            self.main.mode = 5
        elif self.main.ui.YCV_comboBox1.currentText() == "VOC" and self.main.ui.YCV_comboBox2.currentText() == "Yolo":
            self.main.mode = 6
        else :
            self.main.mode = 0

    def Transform(self):
        print(self.main.mode)
        if self.main.mode == 1:
            from utils.YOLO2COCO import yolo_to_coco_converter
            if self.main.Yolo_imagesPath != '' or self.main.Yolo_labelsPath != '' or self.main.Yolo_classPath != '' or self.main.COCO_RootPath != '':
                yolo_to_coco_converter(self.main.Yolo_imagesPath, self.main.Yolo_labelsPath, self.main.Yolo_classPath, self.main.COCO_RootPath)
                QMessageBox.information(self.main, "提示", "转换完成", QMessageBox.Ok)
            else:
                QMessageBox.information(self.main, "提示", "请填写完整信息", QMessageBox.Ok)

        elif self.main.mode == 2:
            from utils.YOLO2VOC import convert_yolo_to_voc
            if self.main.Yolo_imagesPath != '' or self.main.Yolo_labelsPath != '' or self.main.Yolo_classPath != '' or self.main.VOC_RootPath != '':
                convert_yolo_to_voc(self.main.Yolo_imagesPath, self.main.Yolo_labelsPath,self.main.Yolo_classPath, self.main.VOC_RootPath)
                QMessageBox.information(self.main, "提示", "转换完成", QMessageBox.Ok)
            else:
                QMessageBox.information(self.main, "提示", "请填写完整信息", QMessageBox.Ok)

    def batch_split_dataset(self):
        if self.main.imagesPath != '' and self.main.labelsPath != '' and self.main.out_imagesPath != '' and self.main.out_labelsPath != ''  and self.main.SplitRate != 0:
            from utils.split_dataset import split_dataset
            result = split_dataset(self.main.imagesPath, self.main.labelsPath, self.main.out_imagesPath, self.main.out_labelsPath,self.main.SplitRate)
            QMessageBox.information(self.main, "提示", "划分完成", QMessageBox.Ok)

        elif self.main.imagesPath == '' or self.main.labelsPath == '' or self.main.out_imagesPath == '' or self.main.out_labelsPath == '':
            QMessageBox.information(self.main, "提示", "images,labels,out_images和out_labels路径不能为空", QMessageBox.Ok)
        elif self.main.SplitRate  == 0 :
            QMessageBox.information(self.main, "提示", "划分比例文件不能为0", QMessageBox.Ok)

    def batch_rename(self):
        if self.main.imagesPath != '' or self.main.labelsPath != '' or self.main.jsonsPath != '' or self.main.SecondName != '' :
            from utils.rename import rename_files
            rename_files(self.main.imagesPath, self.main.labelsPath, self.main.jsonsPath, self.main.FirstName,self.main.SecondName)
            QMessageBox.information(self.main, "提示", "重命名完成", QMessageBox.Ok)
        elif self.main.imagesPath == '' and self.main.labelsPath == '' and self.main.jsonsPath == '':
            QMessageBox.information(self.main, "提示", "至少选择一个文件路径", QMessageBox.Ok)

    def batch_move(self):
        if self.main.OriginFloder != '' and self.main.TargetFloder != '':
            from utils.move_copy import move_copy
            move_copy(self.main.OriginFloder, self.main.TargetFloder,1,self.main.current_suffix)
        elif self.main.OriginFloder == '' or self.main.TargetFloder == '':
            QMessageBox.information(self.main, "提示", "路径不能为空", QMessageBox.Ok)

    def batch_copy(self):
        if self.main.OriginFloder != '' and self.main.TargetFloder != '':
            from utils.move_copy import move_copy
            move_copy(self.main.OriginFloder, self.main.TargetFloder,0,self.main.current_suffix)
        elif self.main.OriginFloder == '' or self.main.TargetFloder == '':
            QMessageBox.information(self.main, "提示", "路径不能为空", QMessageBox.Ok)

    def Set_Suffix(self):
        # 统一通过lineEdit更新对应变量
        if self.main.ui.comboBox_Suffix.currentText() == "None":
            self.main.current_suffix = None
        else:
            self.main.current_suffix = [self.main.ui.comboBox_Suffix.currentText()]


    def json2txt(self):
        if self.main.jsonsPath != '' and self.main.labelsPath != '' and self.main.classPath != '':
            from utils.json2txt import batch_convert
            batch_convert(self.main.jsonsPath, self.main.labelsPath, self.main.classPath)
        elif self.main.jsonsPath == '' or self.main.labelsPath == '':
            QMessageBox.information(self.main, "提示", "jsons和labels路径不能为空", QMessageBox.Ok)
        elif self.main.classPath == '':
            QMessageBox.information(self.main, "提示", "class文件不能为空", QMessageBox.Ok)

    def txt2json(self):
        if self.main.imagesPath != '' and self.main.labelsPath != '' and self.main.jsonsPath != '' and self.main.classPath != '':
            from utils.txt2json import batch_convert_yolo_to_labelme
            batch_convert_yolo_to_labelme(self.main.imagesPath,self.main.labelsPath, self.main.jsonsPath, self.main.classPath)
        elif self.main.imagesPath == '' or self.main.jsonsPath == '' or self.main.labelsPath == '':
            QMessageBox.information(self.main, "提示", "images,labels和jsons路径不能为空", QMessageBox.Ok)
        elif self.main.classPath == '':
            QMessageBox.information(self.main, "提示", "class文件不能为空", QMessageBox.Ok)

    def SplitByType(self):
        if self.main.imagesPath != '':
            from utils.SplitByTypes import SplitByTypes
            SplitByTypes(self.main.imagesPath)

    def batch_resize(self):
        if self.main.imagesPath != '' and self.main.labelsPath != '' and self.main.ImageSize != 0:
            from utils.resize import yolov8_style_resize
            yolov8_style_resize(self.main.imagesPath, self.main.labelsPath,self.main.ImageSize)
        elif self.main.imagesPath == '' or self.main.labelsPath == '' and self.main.ImageSize != 0 :
            QMessageBox.information(self.main, "提示", "images和labels路径不能为空", QMessageBox.Ok)
        else:
            QMessageBox.information(self.main, "提示", "ImageSize不能为0", QMessageBox.Ok)

    def Set_Enhmode(self):
        if self.main.ui.comboBox_FeatureEnhance.currentText() == "增强图像高频特征" :
            self.main.Enhmode = 1
        elif self.main.ui.comboBox_FeatureEnhance.currentText() == "均衡光照" :
            self.main.Enhmode = 2
        elif self.main.ui.comboBox_FeatureEnhance.currentText() == "融合不同尺度" :
            self.main.Enhmode = 3
        else :
            self.main.Enhmode = 0

    def FeatureEnhance(self):
        if self.main.imagesPath != '' and self.main.out_imagesPath != '':
            from utils.enhance import enhance_high_frequency,normalize_illumination,multi_scale_fusion
            if self.main.Enhmode == 1:
                enhance_high_frequency(self.main.imagesPath, self.main.out_imagesPath)
                QMessageBox.information(self.main, "提示", "增强图像高频特征完成", QMessageBox.Ok)
            elif self.main.Enhmode == 2:
                normalize_illumination(self.main.imagesPath, self.main.out_imagesPath)
                QMessageBox.information(self.main, "提示", "均衡光照完成", QMessageBox.Ok)
            elif self.main.Enhmode == 3:
                multi_scale_fusion(self.main.imagesPath, self.main.out_imagesPath)
                QMessageBox.information(self.main, "提示", "融合不同尺度完成", QMessageBox.Ok)
            else:
                return

        elif self.main.imagesPath == '' or self.main.labelsPath == '' or self.main.out_imagesPath == '' or self.main.out_labelsPath == '':
            QMessageBox.information(self.main, "提示", "images和out_images路径不能为空", QMessageBox.Ok)

    def Set_Augmode(self):
        if self.main.ui.comboBox_DataAugment.currentText() == "旋转翻转平移" :
            self.main.Augmode = 1
        elif self.main.ui.comboBox_DataAugment.currentText() == "图像色彩增广" :
            self.main.Augmode = 2
        elif self.main.ui.comboBox_DataAugment.currentText() == "混合增广" :
            self.main.Augmode = 3
        else :
            self.main.Augmode = 0

    def DataAugment(self):
        if self.main.imagesPath != '' and self.main.labelsPath != '' and self.main.out_imagesPath != '' and self.main.out_labelsPath != '':
            from utils.augment import geometric_augmentation,color_augmentation,advanced_mixing
            if self.main.Augmode == 1:
                geometric_augmentation(self.main.imagesPath, self.main.labelsPath, self.main.out_imagesPath, self.main.out_labelsPath)
                QMessageBox.information(self.main, "提示", "旋转翻转平移增广完成", QMessageBox.Ok)
            elif self.main.Augmode == 2:
                color_augmentation(self.main.imagesPath, self.main.labelsPath, self.main.out_imagesPath, self.main.out_labelsPath)
                QMessageBox.information(self.main, "提示", "图像色彩增广完成", QMessageBox.Ok)
            elif self.main.Augmode == 3:
                advanced_mixing(self.main.imagesPath, self.main.labelsPath, self.main.out_imagesPath, self.main.out_labelsPath)
                QMessageBox.information(self.main, "提示", "混合增广完成", QMessageBox.Ok)
            else:
                return

        elif self.main.imagesPath == '' or self.main.labelsPath == '' or self.main.out_imagesPath == '' or self.main.out_labelsPath == '':
            QMessageBox.information(self.main, "提示", "images,labels,out_images和out_labels路径不能为空", QMessageBox.Ok)

    def Visual(self):
        if self.main.Yolo_imagesPath != '' and self.main.Yolo_labelsPath != '' and self.main.Yolo_classPath != '' and self.main.OutputVisPath != '':
            from utils.VIS import visualize_yolo_labels
            visualize_yolo_labels(self.main.Yolo_imagesPath, self.main.Yolo_labelsPath, self.main.Yolo_classPath, self.main.OutputVisPath)
        if self.main.COCO_trainPath != '' and self.main.COCO_valPath != '' and self.main.COCO_annotationsPath != '' and self.main.OutputVisPath != '':
            from utils.VIS import visualize_coco_full
            visualize_coco_full(self.main.COCO_trainPath, self.main.COCO_valPath, self.main.COCO_annotationsPath, self.main.OutputVisPath)
        if self.main.VOC_AnnotationsPath != '' and self.main.VOC_JPEGImagesPath != '' and self.main.VOC_ImageSetsPath != '' and self.main.OutputVisPath != '':
            from utils.VIS import visualize_voc_dataset
            visualize_voc_dataset(self.main.VOC_AnnotationsPath, self.main.VOC_JPEGImagesPath, self.main.VOC_ImageSetsPath, self.main.OutputVisPath)