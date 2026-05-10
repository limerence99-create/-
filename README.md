# 广西景区智能识别系统 

## 📖 项目简介
本项目是基于深度学习的景区景点智能识别系统，实现了景区图像的自动分类与识别功能。项目采用 ResNet101 作为主干网络进行图像特征提取，并结合 YOLOv8 实现目标检测，最终通过 Web 端和桌面端两种方式提供交互服务。

## 🛠️ 技术栈
- **编程语言**: Python 3.x
- **深度学习框架**: PyTorch
- **模型**: ResNet101（图像分类）、YOLOv8（目标检测）
- **Web 框架**: Flask
- **前端技术**: HTML/CSS
- **桌面端**: PyQt

## 📂 项目结构
.├── core/ # 核心模型与工具代码│ ├── model.py # 模型定义│ └── utils.py # 工具函数├── web/ # Web 端应用│ ├── app_web.py # Flask 服务端│ └── static/│ └── style.css # 前端样式│ └── templates/│ └── index.html # 前端页面├── app_desktop.py # 桌面端应用入口├── app-spotdetect.py # 景点检测主程序├── spot_resnet_train.py # ResNet 模型训练脚本├── spotdataset_make.py # 数据集制作脚本├── writespotlabel.py # 标签处理脚本├── spot_labels.txt # 景点标签文件└── .gitignore # Git 忽略配置
plaintext

## 🚀 运行方式
### 1. 环境安装
```bash
pip install -r requirements.txt
2. 运行 Web 端
bash
运行
cd web
python app_web.py
# 访问 http://127.0.0.1:5000 即可使用
3. 运行桌面端
bash
运行
python app_desktop.py
✨ 功能说明
图像识别: 上传景区图片，自动识别景点类别并输出结果
模型训练: 提供完整的 ResNet 模型训练流程，支持自定义数据集
双端交互: 同时支持 Web 浏览器和桌面端两种使用方式
数据处理: 包含数据集制作、标签处理等完整数据预处理工具
