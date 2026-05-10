import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import os

# 1. 数据预处理与加载
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 适合景点图片的增强
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载景点数据集
data_dir = './spotsdata/spot_dataset/'
full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)

# 自动获取类别总数
num_classes = len(full_dataset.classes)
print(f"数据集包含 {num_classes} 个景点类别: {full_dataset.classes}")

# 划分数据集
dataset_size = len(full_dataset)
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# 应用验证集变换
val_dataset.dataset.transform = val_transform

# 创建数据加载器
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


# 2. 创建ResNet101模型
def create_resnet101_model(num_classes):
    model = models.resnet101(pretrained=True)

    # 冻结初始层
    for param in model.parameters():
        param.requires_grad = False

    # 替换全连接层
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )

    return model


# 初始化模型、损失函数和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_resnet101_model(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.001
)

# 3. 训练循环
num_epochs = 30
best_val_accuracy = 0.0

print("开始训练景点识别模型...")
print(f"设备: {device}")

for epoch in range(num_epochs):
    # 第10轮后解冻所有层进行微调
    if epoch == 10:
        print("解冻所有层，开始微调整个网络...")
        for param in model.parameters():
            param.requires_grad = True
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=0.0001
        )

    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader.dataset)
    train_accuracy = 100 * correct_predictions / total_samples

    # 验证阶段
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_accuracy = 100 * correct_val / total_val
    avg_val_loss = val_loss / len(val_loader.dataset)

    phase = "全连接层训练" if epoch < 10 else "全网络微调"
    print(f'Epoch [{epoch + 1:2d}/{num_epochs}] ({phase}): ')
    print(f'  训练损失: {epoch_loss:.4f}, 训练准确率: {train_accuracy:.2f}%')
    print(f'  验证损失: {avg_val_loss:.4f}, 验证准确率: {val_accuracy:.2f}%')

    # 保存最佳模型
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save({
            'model_state_dict': model.state_dict(),
            'class_to_idx': full_dataset.class_to_idx,
            'num_classes': num_classes,
            'model_architecture': 'resnet101'
        }, 'spot_recognition_resnet101.pth')
        print(f'  保存最佳模型，验证准确率: {val_accuracy:.2f}%')

print(f"\n训练完成！最佳验证准确率: {best_val_accuracy:.2f}%")
print("模型已保存为 'spot_recognition_resnet101.pth'！")