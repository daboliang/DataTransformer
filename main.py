from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QComboBox

from Slotfunc import SlotFunctions  # 新增导入

from mainWin import Ui_MainWindow

class DataTransformer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # 初始化变量
        ######## 页面1 #########
        self.Yolo_RootPath = ''
        self.Yolo_imagesPath = ''
        self.Yolo_labelsPath = ''
        self.Yolo_classPath = ''

        self.COCO_RootPath = ''
        self.COCO_trainPath = ''
        self.COCO_valPath = ''
        self.COCO_annotationsPath = ''

        self.VOC_RootPath = ''
        self.VOC_AnnotationsPath = ''
        self.VOC_JPEGImagesPath = ''
        self.VOC_ImageSetsPath = ''

        # 转换模式
        self.mode = 1
        self.OutputVisPath = ''


        ######## 页面2 #########

        self.imagesPath = ''
        self.labelsPath = ''
        self.jsonsPath = ''
        self.out_imagesPath = ''
        self.out_labelsPath = ''
        self.OriginFloder = ''
        self.TargetFloder = ''
        self.SplitRate = 0
        self.FirstName = 0
        self.SecondName = ''
        self.ImageSize = 0
        self.Suffix = []
        self.current_suffix = None
        self.classPath = ''

        self.Enhmode = 0
        self.Augmode = 0

        # 创建槽函数处理器实例
        self.slot_handler = SlotFunctions(self)
        self.connect_actions()
        self.slot_handler.Set_mode()



    def connect_actions(self):
        """连接所有按钮信号与槽函数"""
        # 页面切换（保持原状）
        self.ui.Data_Transform_Btn.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(0))
        self.ui.Data_Batch_Btn.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(1))
        # 文件操作
        #############     Yolo      ###############
        self.ui.Yolo_RootPath_Btn.clicked.connect(lambda: self.slot_handler.load_Path(self.ui.Yolo_RootPath_lineEdit))
        self.ui.Yolo_imagesPath_Btn.clicked.connect(lambda:self.slot_handler.load_Path(self.ui.Yolo_imagesPath_lineEdit))
        self.ui.Yolo_labelsPath_Btn.clicked.connect(lambda:self.slot_handler.load_Path(self.ui.Yolo_labelsPath_lineEdit))
        self.ui.Yolo_classPath_Btn.clicked.connect(lambda:self.slot_handler.load_FilePath(self.ui.Yolo_classPath_lineEdit))

        self.ui.Yolo_RootPath_lineEdit.textChanged.connect(lambda: self.slot_handler.set_txt(self.ui.Yolo_RootPath_lineEdit))
        self.ui.Yolo_imagesPath_lineEdit.textChanged.connect(lambda:self.slot_handler.set_txt(self.ui.Yolo_imagesPath_lineEdit))
        self.ui.Yolo_labelsPath_lineEdit.textChanged.connect(lambda:self.slot_handler.set_txt(self.ui.Yolo_labelsPath_lineEdit))
        self.ui.Yolo_classPath_lineEdit.textChanged.connect(lambda:self.slot_handler.set_txt(self.ui.Yolo_classPath_lineEdit))

        #############     COCO      ###############
        self.ui.COCO_RootPath_Btn.clicked.connect(lambda: self.slot_handler.load_Path(self.ui.COCO_RootPath_lineEdit))
        self.ui.COCO_trainPath_Btn.clicked.connect(lambda: self.slot_handler.load_Path(self.ui.COCO_trainPath_lineEdit))
        self.ui.COCO_valPath_Btn.clicked.connect(lambda: self.slot_handler.load_Path(self.ui.COCO_valPath_lineEdit))
        self.ui.COCO_annotationsPath_Btn.clicked.connect(lambda: self.slot_handler.load_Path(self.ui.COCO_annotationsPath_lineEdit))

        self.ui.COCO_RootPath_lineEdit.textChanged.connect(lambda: self.slot_handler.set_txt(self.ui.COCO_RootPath_lineEdit))
        self.ui.COCO_trainPath_lineEdit.textChanged.connect(lambda: self.slot_handler.set_txt(self.ui.COCO_trainPath_lineEdit))
        self.ui.COCO_valPath_lineEdit.textChanged.connect(lambda: self.slot_handler.set_txt(self.ui.COCO_valPath_lineEdit))
        self.ui.COCO_annotationsPath_lineEdit.textChanged.connect(lambda: self.slot_handler.set_txt(self.ui.COCO_annotationsPath_lineEdit))

        #############     VOC      ###############
        self.ui.VOC_RootPath_Btn.clicked.connect(lambda: self.slot_handler.load_Path(self.ui.VOC_RootPath_lineEdit))
        self.ui.VOC_AnnotationsPath_Btn.clicked.connect(lambda: self.slot_handler.load_Path(self.ui.VOC_AnnotationsPath_lineEdit))
        self.ui.VOC_JPEGImagesPat_Btn.clicked.connect(lambda: self.slot_handler.load_Path(self.ui.VOC_JPEGImagesPath_lineEdit))
        self.ui.VOC_ImageSetsPath_Btn.clicked.connect(lambda: self.slot_handler.load_Path(self.ui.VOC_ImageSetsPath_lineEdit))

        self.ui.VOC_RootPath_lineEdit.textChanged.connect(lambda: self.slot_handler.set_txt(self.ui.VOC_RootPath_lineEdit))
        self.ui.VOC_AnnotationsPath_lineEdit.textChanged.connect(lambda: self.slot_handler.set_txt(self.ui.VOC_AnnotationsPath_lineEdit))
        self.ui.VOC_JPEGImagesPath_lineEdit.textChanged.connect(lambda: self.slot_handler.set_txt(self.ui.VOC_JPEGImagesPath_lineEdit))
        self.ui.VOC_ImageSetsPath_lineEdit.textChanged.connect(lambda: self.slot_handler.set_txt(self.ui.VOC_ImageSetsPath_lineEdit))

        # 转换函数
        self.ui.Transform_Btn.clicked.connect(lambda: self.slot_handler.Transform())
        self.ui.YCV_comboBox1.addItems(["Yolo", "COCO", "VOC"])  # 批量添加
        self.ui.YCV_comboBox2.addItems(["Yolo", "COCO", "VOC"])  # 批量添加
        self.ui.YCV_comboBox1.setCurrentIndex(0)
        self.ui.YCV_comboBox1.currentIndexChanged.connect(lambda: self.slot_handler.Set_mode())
        self.ui.YCV_comboBox2.setCurrentIndex(1)
        self.ui.YCV_comboBox2.currentIndexChanged.connect(lambda: self.slot_handler.Set_mode())

        self.ui.pushButton_Visual.clicked.connect(lambda: self.slot_handler.Visual())
        self.ui.pushButton_OutputVisPath.clicked.connect(lambda: self.slot_handler.load_Path(self.ui.OutputVisPath_lineEdit))
        self.ui.OutputVisPath_lineEdit.textChanged.connect(lambda: self.slot_handler.set_txt(self.ui.OutputVisPath_lineEdit))


        #############     batch handler      ###############
        self.ui.Btn_images.clicked.connect(lambda: self.slot_handler.load_Path(self.ui.imagesPath_lineEdit))
        self.ui.Btn_labels.clicked.connect(lambda: self.slot_handler.load_Path(self.ui.labelsPath_lineEdit))
        self.ui.Btn_jsons.clicked.connect(lambda: self.slot_handler.load_Path(self.ui.jsonsPath_lineEdit))
        self.ui.Btn_out_images.clicked.connect(lambda: self.slot_handler.load_Path(self.ui.out_imagesPath_lineEdit))
        self.ui.Btn_out_labels.clicked.connect(lambda: self.slot_handler.load_Path(self.ui.out_labelsPath_lineEdit))
        self.ui.pushButton_classPath.clicked.connect(lambda: self.slot_handler.load_FilePath(self.ui.classPath_lineEdit))

        self.ui.imagesPath_lineEdit.textChanged.connect(lambda: self.slot_handler.set_txt(self.ui.imagesPath_lineEdit))
        self.ui.labelsPath_lineEdit.textChanged.connect(lambda: self.slot_handler.set_txt(self.ui.labelsPath_lineEdit))
        self.ui.jsonsPath_lineEdit.textChanged.connect(lambda: self.slot_handler.set_txt(self.ui.jsonsPath_lineEdit))
        self.ui.classPath_lineEdit.textChanged.connect(lambda: self.slot_handler.set_txt(self.ui.classPath_lineEdit))
        self.ui.out_imagesPath_lineEdit.textChanged.connect(lambda: self.slot_handler.set_txt(self.ui.out_imagesPath_lineEdit))
        self.ui.out_labelsPath_lineEdit.textChanged.connect(lambda: self.slot_handler.set_txt(self.ui.out_labelsPath_lineEdit))

        self.ui.doubleSpinBox_SplitRate.setValue(0)
        self.ui.doubleSpinBox_SplitRate.setSingleStep(0.1)
        self.ui.doubleSpinBox_SplitRate.setMaximum(1)
        self.ui.doubleSpinBox_SplitRate.setMinimum(0)
        self.ui.doubleSpinBox_SplitRate.valueChanged.connect(lambda: self.slot_handler.set_float(self.ui.doubleSpinBox_SplitRate))
        self.ui.pushButton_split.clicked.connect(lambda: self.slot_handler.batch_split_dataset())

        self.ui.spinBox_ImageSize.setValue(0)
        self.ui.spinBox_ImageSize.setMaximum(2000)
        self.ui.spinBox_ImageSize.textChanged.connect(lambda: self.slot_handler.set_int(self.ui.spinBox_ImageSize))
        self.ui.pushButton_resize.clicked.connect(lambda: self.slot_handler.batch_resize())

        self.ui.spinBox_FirstName.textChanged.connect(lambda: self.slot_handler.set_int(self.ui.spinBox_FirstName))
        self.ui.SecondName_lineEdit.textChanged.connect(lambda: self.slot_handler.set_txt(self.ui.SecondName_lineEdit))
        self.ui.pushButton_rename.clicked.connect(lambda: self.slot_handler.batch_rename())

        self.ui.pushButton_OriginFloder.clicked.connect(lambda: self.slot_handler.load_Path(self.ui.OriginFloder_lineEdit))
        self.ui.OriginFloder_lineEdit.textChanged.connect(lambda: self.slot_handler.set_txt(self.ui.OriginFloder_lineEdit))
        self.ui.pushButton_TargetFloder.clicked.connect(lambda: self.slot_handler.load_Path(self.ui.TargetFloder_lineEdit))
        self.ui.TargetFloder_lineEdit.textChanged.connect(lambda: self.slot_handler.set_txt(self.ui.TargetFloder_lineEdit))
        self.ui.pushButton_Move.clicked.connect(lambda: self.slot_handler.batch_move())
        self.ui.pushButton_Copy.clicked.connect(lambda: self.slot_handler.batch_copy())
        self.ui.comboBox_Suffix.currentIndexChanged.connect(lambda: self.slot_handler.Set_Suffix())

        self.ui.pushButton_json2txt.clicked.connect(lambda: self.slot_handler.json2txt())
        self.ui.pushButton_txt2json.clicked.connect(lambda: self.slot_handler.txt2json())

        self.ui.pushButton_SplitByType.clicked.connect(lambda: self.slot_handler.SplitByType())

        self.ui.comboBox_FeatureEnhance.addItems(["None", "增强图像高频特征", "均衡光照", "融合不同尺度"])
        self.ui.comboBox_FeatureEnhance.setCurrentIndex(0)
        self.ui.comboBox_FeatureEnhance.currentIndexChanged.connect(lambda: self.slot_handler.Set_Enhmode())
        self.ui.pushButton_FeatureEnhance.clicked.connect(lambda: self.slot_handler.FeatureEnhance())

        self.ui.comboBox_DataAugment.addItems(["None", "旋转翻转平移", "图像色彩增广", "混合增广"])
        self.ui.comboBox_DataAugment.setCurrentIndex(0)
        self.ui.comboBox_DataAugment.currentIndexChanged.connect(lambda: self.slot_handler.Set_Augmode())
        self.ui.pushButton_DataAugmeny.clicked.connect(lambda: self.slot_handler.DataAugment())





if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    window = DataTransformer()
    window.show()
    sys.exit(app.exec_()) 