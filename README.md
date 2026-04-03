本文提出了一种基于YOLOv8目标检测算法与PaddleOCR文字识别技术的新能源车牌检测与识别方法。通过YOLOv8模型实现车牌的精准定位，利用PaddleOCR完成车牌字符的识别，最终开发了基于PySide6的图形用户界面，为用户提供直观的操作体验。在CCPD2020新能源车牌数据集上的实验结果表明，该方法能够有效完成新能源车牌的端到端检测与识别任务。

This paper proposes a new energy vehicle license plate detection and recognition method based on the YOLOv8 object detection algorithm and PaddleOCR optical character recognition technology. The YOLOv8 model is employed to achieve precise localization of license plates, while PaddleOCR is utilized to recognize the characters on the plates. Additionally, a graphical user interface (GUI) based on PySide6 is developed to provide an intuitive user experience. Experimental results on the CCPD2020 new energy license plate dataset demonstrate that the proposed method effectively accomplishes end-to-end detection and recognition of new energy license plates.  

UI使用：点击导入文件，传入新能源车牌图片，点击目标检测，执行YOLO目标检测，将检测到的车牌抠图并显示，点击车牌识别，执行PaddlOCR识别车牌号并显示。

<img width="425" height="384" alt="image" src="https://github.com/user-attachments/assets/4ab9d1ab-d9d4-4bff-8977-73200155c7a4" />
<img width="425" height="385" alt="image" src="https://github.com/user-attachments/assets/97f5975c-d283-4cf7-aecb-8692149ad204" />
