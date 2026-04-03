from paddleocr import TextRecognition

model = TextRecognition()
output = model.predict(
    input=r"D:\Car_yolo_ocr\output\01-90_265-231&522_405&574-405&571_235&574_231&523_403&522-0_0_3_1_28_29_30_30-134-56_obj00_class0.jpg")
print(output)
for res in output:
    res.print()
    res.save_to_json(save_path="./res.json")
