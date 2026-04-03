import torch
from ultralytics import YOLO


def check_environment():
    """检查环境配置"""
    print("=== 环境检查 ===")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")
        print(f"GPU数量: {torch.cuda.device_count()}")
    else:
        print("警告: 未检测到GPU，将使用CPU训练（速度较慢）")

    print(f"Ultralytics版本: {YOLO._version}")


def train_license_plate_model():
    """训练车牌检测模型"""
    print("\n=== 开始训练 ===")

    # 1. 加载模型 (基于PyTorch)
    model = YOLO('yolov11n.pt')

    # 2. 训练模型 (使用GPU)
    results = model.train(
        data='./datasets/CCPD2020/ccpd2020_green.yaml',
        epochs=5,
        imgsz=640,
        batch=64,
        device=0 if torch.cuda.is_available() else 'cpu',  # 使用GPU 0
        workers=4,
        patience=10,
        name='model_V3'
    )

    print("训练完成!")
    return results


if __name__ == "__main__":
    # 检查环境
    check_environment()

    # 开始训练
    train_license_plate_model()
