import os
from pathlib import Path
import cv2
from ultralytics import YOLO
import argparse


def create_spot_dataset(source_root, output_root, model_weights='yolov8n.pt', conf_threshold=0.5):
    """
    使用YOLOv8目标检测模型从图片中提取景点区域，创建景点数据集。
    保留完整场景图片（无需人脸检测逻辑）
    """

    # 加载YOLOv8模型
    model = YOLO(model_weights)
    print(f"已加载模型: {model_weights}")

    # 创建输出目录
    Path(output_root).mkdir(parents=True, exist_ok=True)

    # 支持处理的图片格式
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

    # 遍历源目录中的每个子目录（每个景点的名字）
    for spot_name in os.listdir(source_root):
        spot_source_dir = os.path.join(source_root, spot_name)
        spot_output_dir = os.path.join(output_root, spot_name)

        # 检查是否为目录
        if not os.path.isdir(spot_source_dir):
            continue

        # 为当前景点创建输出目录
        Path(spot_output_dir).mkdir(parents=True, exist_ok=True)
        print(f"\n正在处理景点: {spot_name}")
        print(f"源目录: {spot_source_dir}")
        print(f"输出目录: {spot_output_dir}")

        image_files = [f for f in os.listdir(spot_source_dir)
                       if f.lower().endswith(image_extensions)]

        processed_count = 0
        skipped_count = 0

        # 处理当前景点的每张图片
        for image_file in image_files:
            image_path = os.path.join(spot_source_dir, image_file)

            # 使用OpenCV读取图片
            image = cv2.imread(image_path)
            if image is None:
                print(f"  跳过无法读取的图片: {image_file}")
                skipped_count += 1
                continue

            # 景点识别保留完整图片，仅进行尺寸标准化
            resized_image = cv2.resize(image, (640, 480))  # 统一尺寸

            # 保存处理后的图片
            output_path = os.path.join(spot_output_dir, image_file)
            cv2.imwrite(output_path, resized_image)
            processed_count += 1

        print(f"  完成！处理图片: {len(image_files)}张, 成功保存: {processed_count}张, 跳过: {skipped_count}张")

    print(f"\n数据集创建完成！输出路径: {output_root}")


if __name__ == "__main__":
    # 参数设置
    source_directory = "./spotsdata/dataset"  # 原始景点图片根目录
    output_directory = "./spotsdata/spot_dataset"  # 输出数据集位置

    create_spot_dataset(source_directory, output_directory)