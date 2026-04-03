# test_on_testset.py
from ultralytics import YOLO
import torch
from pathlib import Path
import cv2


def test_with_best_model():
    """使用最佳模型在测试集上进行检测"""

    print("=== 测试集检测 ===")

    # 1. 加载训练好的最佳模型
    model_path = 'runs/detect/model_V2/weights/best.pt'

    if not Path(model_path).exists():
        print(f"❌ 模型文件不存在: {model_path}")
        print("🔍 寻找其他模型文件...")

        # 查找可能的模型文件
        model_files = list(Path('.').glob('**/best.pt'))
        if model_files:
            model_path = str(model_files[0])
            print(f"✅ 找到模型: {model_path}")
        else:
            print("❌ 没有找到训练好的模型")
            return

    print(f"📁 加载模型: {model_path}")
    model = YOLO(model_path)

    # 2. 测试集路径
    test_images_dir = Path('./datasets/CCPD2020/images/test')

    if not test_images_dir.exists():
        print(f"❌ 测试集目录不存在: {test_images_dir}")
        print("🔍 寻找测试集...")

        # 尝试其他可能的测试集路径
        possible_paths = [
            './datasets/CCPD2020/images/val',
            './datasets/CCPD2020/images/train',
            './datasets/CCPD2020/Bm test',  # 你之前的目录结构
        ]

        for path in possible_paths:
            if Path(path).exists():
                test_images_dir = Path(path)
                print(f"✅ 使用替代路径: {test_images_dir}")
                break
        else:
            print("❌ 没有找到测试集")
            return

    # 3. 执行检测
    print(f"🔍 检测测试集: {test_images_dir}")

    results = model.predict(
        source=str(test_images_dir),
        save=True,  # 保存检测结果图片
        save_txt=True,  # 保存检测结果标签
        device=0 if torch.cuda.is_available() else 'cpu',  # 使用GPU 0
        save_conf=True,  # 保存置信度
        conf=0.25,  # 置信度阈值
        iou=0.45,  # IoU阈值
        show=False,  # 不显示图片（批量处理时）
        project='runs/detect',  # 结果保存目录
        name='test_results_V2',  # 实验名称
        exist_ok=True  # 覆盖已存在的结果
    )

    print("✅ 测试集检测完成!")

    return results


def test_single(model_path, data_path):
    # 1. 加载训练好的最佳模型
    model = YOLO(model_path)

    # 3. 执行检测
    print(f"🔍 检测测试集: {data_path}")
    results = model.predict(
        source=str(data_path),
        conf=0.25,  # 置信度阈值
        iou=0.45,  # IoU阈值
        show=True,  # 显示图片（批量处理时）
        save=True,  # 保存检测结果图片
        save_txt=True,  # 保存检测结果标签
        exist_ok=True  # 覆盖已存在的结果
    )

    return results


def create_from_yolo_label_file(data_path, img_path):
    """
    将yolo识别结果转化为坐标格式
    """
    result_list = []
    labels_path = Path(data_path)
    images_path = Path(img_path)
    output_path = Path('D:/Car_yolo_ocr/output')
    output_path.mkdir(parents=True, exist_ok=True)

    # 创建文件名（无后缀）到文件路径的映射
    label_files = {f.stem: f for f in labels_path.glob("*.txt")}
    image_files = {f.stem: f for f in images_path.glob("*.jpg")}

    # 找到共同的文件名（无后缀）
    common_files = set(label_files.keys()) & set(image_files.keys())

    print(f"找到 {len(common_files)} 对匹配的文件")

    for file_stem in common_files:
        label_file_path = label_files[file_stem]
        img_file_path = image_files[file_stem]

        # 读取图片获取尺寸
        img = cv2.imread(str(img_file_path))
        if img is None:
            print(f"警告：无法读取图像 {img_file_path}，跳过")
            continue

        # 读取图片的高和宽
        img_height, img_width = img.shape[:2]

        # 打开txt文件，读取所有行
        with open(label_file_path, 'r') as f:
            lines = f.readlines()

        object_count = 0
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue

            # 以空格符分割字符串
            answer = line.split(" ")

            # 验证数据完整性
            if len(answer) < 5:
                print(f"警告：{file_stem} 第{line_num}行数据不完整，跳过")
                continue

            try:
                # 转为float变量
                class_id = int(answer[0])  # 类别ID
                x_center = float(answer[1])
                y_center = float(answer[2])
                anchor_width = float(answer[3])
                anchor_height = float(answer[4])

                # 将yolo变量转为int类型图像坐标变量
                x_center_pixel = x_center * img_width
                y_center_pixel = y_center * img_height
                anchor_width_pixel = anchor_width * img_width
                anchor_height_pixel = anchor_height * img_height

                x1 = int(x_center_pixel - anchor_width_pixel / 2.0)
                y1 = int(y_center_pixel - anchor_height_pixel / 2.0)
                x2 = int(x_center_pixel + anchor_width_pixel / 2.0)
                y2 = int(y_center_pixel + anchor_height_pixel / 2.0)

                # 边界检查
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(img_width - 1, x2)
                y2 = min(img_height - 1, y2)

                # 确保裁剪区域有效
                if x2 <= x1 or y2 <= y1:
                    print(f"警告：{file_stem} 第{line_num}行坐标无效，跳过")
                    continue

                # 存储为坐标点
                anchor_point = {
                    'file_stem': file_stem,
                    'object_index': object_count,
                    'class_id': class_id,
                    'coordinates': [(y1, x1), (y2, x2)],
                    'pixel_coords': (x1, y1, x2, y2)
                }

                # 存储在结果列表中
                result_list.append(anchor_point)

                # 裁剪图像
                cropped = img[y1:y2, x1:x2]

                # 生成唯一文件名
                output_filename = f"{file_stem}_obj{object_count:02d}_class{class_id}.jpg"
                output_file_path = output_path / output_filename

                cv2.imwrite(str(output_file_path), cropped)

                print(f"===== 完成 {file_stem} 第 {object_count + 1} 个对象的转换 =====")
                print(f"坐标: ({x1}, {y1}) - ({x2}, {y2})")
                print(f"***** 保存为: {output_filename} *****")

                object_count += 1

            except (ValueError, IndexError) as e:
                print(f"错误：处理 {file_stem} 第{line_num}行时出错: {e}")
                continue

        print(f"=== 完成图像 {file_stem} 的处理，共 {object_count} 个对象 ===\n")

    # 返回结果
    return result_list


def single_img_process(label_files, image_files):
    label_file_path = Path(label_files)
    img_file_path = Path(image_files)
    output_path = Path('D:/Car_yolo_ocr/yolo_answer')
    output_path.mkdir(parents=True, exist_ok=True)

    # 读取图片获取尺寸
    img = cv2.imread(str(img_file_path))
    if img is None:
        print(f"警告：无法读取图像 {img_file_path}，跳过")

    # 读取图片的高和宽
    img_height, img_width = img.shape[:2]

    # 打开txt文件，读取所有行
    with open(label_file_path, 'r') as f:
        lines = f.readline()

        # 以空格符分割字符串
        answer = lines.strip("\n").split(" ")

        # 转为float变量
        x_center = float(answer[1])
        y_center = float(answer[2])
        anchor_width = float(answer[3])
        anchor_height = float(answer[4])

        # 将yolo变量转为int类型图像坐标变量
        x_center_pixel = x_center * img_width
        y_center_pixel = y_center * img_height
        anchor_width_pixel = anchor_width * img_width
        anchor_height_pixel = anchor_height * img_height

        x1 = int(x_center_pixel - anchor_width_pixel / 2.0)
        y1 = int(y_center_pixel - anchor_height_pixel / 2.0)
        x2 = int(x_center_pixel + anchor_width_pixel / 2.0)
        y2 = int(y_center_pixel + anchor_height_pixel / 2.0)

        # 边界检查
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_width - 1, x2)
        y2 = min(img_height - 1, y2)

        # 存储为坐标点
        anchor_point = {
            'coordinates': [(y1, x1), (y2, x2)],
            'pixel_coords': (x1, y1, x2, y2)
        }

        # 裁剪图像
        cropped = img[y1:y2, x1:x2]

        # 生成唯一文件名
        output_filename = f"{img_file_path.stem}_cropped.jpg"
        output_file_path = output_path / output_filename

        cv2.imwrite(str(output_file_path), cropped)

    return output_file_path


if __name__ == "__main__":
    # test_with_best_model()

    model_path_str = r"D:\Car_yolo_ocr\runs\detect\model_V2\weights\best.pt"
    data_path_str = r"D:\Car_yolo_ocr\03-103_253-267&425_483&565-483&565_271&497_267&425_480&483-0_0_3_25_25_33_25_25-110-47.jpg"
    result = test_single(model_path_str, data_path_str)
    print(result)

    # img_path = Path(r"D:\Car_yolo_ocr\runs\detect\test_results_V1")
    # path = img_path / 'labels'
    # result = create_from_yolo_label_file(path, img_path)
    # print(result)
