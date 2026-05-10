import torch
from torchvision import datasets, transforms
import os

# 定义数据变换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载数据集
data_dir = './spotsdata/spot_dataset/'
full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# 获取类别信息
num_classes = len(full_dataset.classes)
print(f"数据集包含 {num_classes} 个景点类别: {full_dataset.classes}")

# 将标签保存到TXT文件
with open('spot_labels.txt', 'w', encoding='utf-8') as f:
    for label in full_dataset.classes:
        f.write(label + '\n')

print(f"景点标签已保存到 spot_labels.txt，共 {num_classes} 个标签")