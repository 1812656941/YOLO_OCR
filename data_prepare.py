import cv2
from pathlib import Path


def create_yolo_labels_from_ccpd():
    """
    从CCPD文件名创建YOLO格式的标签文件
    CCPD标注信息藏在文件名中
    """

    dataset_root = Path('./datasets/CCPD2020')
    images_train_dir = dataset_root / 'images' / 'val'
    labels_train_dir = dataset_root / 'labels' / 'val'

    labels_train_dir.mkdir(parents=True, exist_ok=True)

    print("正在从CCPD文件名创建标签...")

    for img_path in images_train_dir.glob('*.jpg'):
        # 解析CCPD文件名获取标注信息
        annotation = parse_ccpd_filename(img_path.name)

        if annotation:
            # 创建对应的标签文件
            label_path = labels_train_dir / f"{img_path.stem}.txt"
            create_yolo_label_file(annotation, label_path, img_path)

    print("标签创建完成!")


def parse_ccpd_filename(filename):
    """解析CCPD文件名获取边界框信息"""
    try:
        # 示例: "025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg"
        name_parts = filename.split('.')[0].split('-')

        if len(name_parts) >= 3:
            # 解析边界框: "154&383_386&473"
            bbox_str = name_parts[2]
            bbox_points = bbox_str.replace('&', '_').split('_')
            bbox = [int(bbox_points[0]), int(bbox_points[1]),
                    int(bbox_points[2]), int(bbox_points[3])]

            return {'bbox': bbox}

    except Exception as e:
        print(f"解析文件名失败 {filename}: {e}")

    return None


def create_yolo_label_file(annotation, label_path, img_path):
    """创建YOLO格式的标签文件"""
    # 读取图片获取尺寸
    img = cv2.imread(str(img_path))
    if img is None:
        return

    img_height, img_width = img.shape[:2]

    # 转换边界框为YOLO格式
    x1, y1, x2, y2 = annotation['bbox']

    x_center = (x1 + x2) / 2 / img_width
    y_center = (y1 + y2) / 2 / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height

    # YOLO格式: class x_center y_center width height
    yolo_line = f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

    with open(label_path, 'w') as f:
        f.write(yolo_line)


if __name__ == "__main__":
    create_yolo_labels_from_ccpd()
