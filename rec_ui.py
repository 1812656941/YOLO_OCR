from PySide6.QtGui import *
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtUiTools import QUiLoader
import cv2
from pathlib import Path
from test import test_single, single_img_process
from paddleocr import TextRecognition
import json


class Stats:
    def __init__(self):
        self.ui = QUiLoader().load('./show.ui')
        self.ui.setWindowTitle("车牌号检测与识别")
        self.ui.open_file.clicked.connect(self.open_file_handle)
        self.ui.yolo_run.clicked.connect(self.yolo_run_handle)
        self.ui.ocr_run.clicked.connect(self.ocr_run_handle)

        self.openfile_path = ""
        self.cap = cv2.VideoCapture()
        self.video_count = 0

    def open_file_handle(self):
        self.video = None
        self.openfile_path = QFileDialog.getOpenFileName()[0]
        video_type = [".mp4", ".mkv", ".MOV", "avi"]
        img_type = [".bmp", ".jpg", ".png", ".gif"]
        for vdi in video_type:
            if vdi not in self.openfile_path:
                continue
            else:
                self.video = True
                # 当是视频时，将开始按钮置为可点击状态
        for ig in img_type:
            if ig not in self.openfile_path:
                continue
            else:
                self.video = False
                self.img = QPixmap(self.openfile_path)
                w = self.img.width()
                h = self.img.height()
                ratio = max(w / self.ui.label.width(), h / self.ui.label.height())
                self.img.setDevicePixelRatio(ratio)
                self.ui.label.setAlignment(Qt.AlignCenter)
                self.ui.label.setPixmap(self.img)
        if self.video is None:
            QMessageBox.information(self, "警告", "我们暂时不支持此格式的文件！", QMessageBox.Ok)

    def show_pic(self):
        self.video_count = self.video_count + 1
        ret, img = self.cap.read()
        if ret:
            self.cur_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # 视频流的长和宽
            height, width = self.cur_frame.shape[:2]
            pixmap = QImage(self.cur_frame, width, height, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(pixmap)
            # 获取是视频流和label窗口的长宽比值的最大值，适应label窗口播放，不然显示不全
            ratio = max(width / self.ui.label.width(), height / self.ui.label.height())
            pixmap.setDevicePixelRatio(ratio)
            # 视频流置于label中间部分播放
            self.ui.label.setAlignment(Qt.AlignCenter)
            self.ui.label.setPixmap(pixmap)

    def yolo_run_handle(self):
        model_path_str = r"./runs\detect\model_V2\weights\best.pt"
        data_path_str = Path(self.openfile_path)
        test_single(model_path_str, data_path_str)

        file_name = Path(r"./runs\detect\predict\labels/" + data_path_str.stem + '.txt')

        self.output_file_path = single_img_process(file_name, data_path_str)

        self.img = QPixmap(self.output_file_path)
        w = self.img.width()
        h = self.img.height()
        ratio = max(w / self.ui.label.width(), h / self.ui.label.height())
        self.img.setDevicePixelRatio(ratio)
        self.ui.label.setAlignment(Qt.AlignCenter)
        self.ui.label.setPixmap(self.img)

    def ocr_run_handle(self):
        model = TextRecognition()
        output = model.predict(input=str(self.output_file_path))
        for res in output:
            res.print()
            res.save_to_json(save_path="./res.json")

        with open('res.json', 'r', encoding='utf-8') as file:
            data = json.load(file)
            self.ui.lineEdit.setText(data['rec_text'])


if __name__ == "__main__":
    app = QApplication()
    stats = Stats()
    stats.ui.show()
    app.exec()
