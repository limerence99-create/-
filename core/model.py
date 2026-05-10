import torch
import torch.nn as nn
from torchvision import models


def load_model():
    try:
        checkpoint = torch.load('spot_recognition_resnet101.pth', map_location='cpu')
        num_classes = checkpoint['num_classes']

        model = models.resnet101(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        model.eval()
        print("\\u2705 模型加载成功")
        return model
    except Exception as e:
        print(f"\\u274c 模型加载失败: {e}")
        raise e