import sys
import os
import torch
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel,
                             QWidget, QFileDialog, QMessageBox, QLineEdit, QTextEdit)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from core.model import load_model
from core.utils import preprocess_image, get_prediction, get_spot_info


class SpotRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = None
        self.spot_labels = []
        self.current_image_path = None
        self.init_ui()
        self.load_model()
        self.load_spot_labels()

    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle('广西景点识别系统')
        self.setFixedSize(1200, 800)

        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局 - 水平布局，分为左右两栏
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        # 左侧区域：图像上传和显示
        left_widget = QWidget()
        left_widget.setStyleSheet('background-color: white; border-radius: 10px; padding: 20px;')
        left_layout = QVBoxLayout()
        left_widget.setLayout(left_layout)

        # 顶部状态栏
        status_bar = QWidget()
        status_bar.setStyleSheet('background-color: #3498db; border-radius: 8px; padding: 10px;')
        status_layout = QHBoxLayout()
        status_bar.setLayout(status_layout)

        # 识别结果按钮
        result_btn = QPushButton('识别结果')
        result_btn.setStyleSheet('''
            background-color: #ffffff;
            color: #3498db;
            border-radius: 20px;
            padding: 5px 15px;
            font-weight: bold;
        ''')

        # 置信度显示
        self.confidence_label = QLabel('置信度: 0.0%')
        self.confidence_label.setStyleSheet('''
            background-color: #ffffff;
            color: #3498db;
            border-radius: 20px;
            padding: 5px 15px;
            font-weight: bold;
        ''')

        status_layout.addWidget(result_btn)
        status_layout.addWidget(self.confidence_label)
        left_layout.addWidget(status_bar)

        # 图像上传区域
        upload_area = QWidget()
        upload_area.setStyleSheet('''
            background-color: #f8f9fa;
            border: 2px dashed #bdc3c7;
            border-radius: 10px;
            padding: 30px;
            min-height: 500px;
        ''')
        upload_layout = QVBoxLayout()
        upload_area.setLayout(upload_layout)

        # 上传图标
        upload_icon = QLabel('↑')
        upload_icon.setAlignment(Qt.AlignCenter)
        upload_icon.setStyleSheet('font-size: 40px; color: #95a5a6;')
        upload_layout.addWidget(upload_icon)

        # 提示文字
        upload_text = QLabel('将图像拖放到此处\n- 或 -\n点击上传')
        upload_text.setAlignment(Qt.AlignCenter)
        upload_text.setStyleSheet('font-size: 14px; color: #7f8c8d; line-height: 1.5;')
        upload_layout.addWidget(upload_text)

        left_layout.addWidget(upload_area)

        # 上传按钮
        self.upload_btn = QPushButton('点击上传')
        self.upload_btn.clicked.connect(self.upload_image)
        self.upload_btn.setStyleSheet('''
            background-color: #e74c3c;
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            font-weight: bold;
            margin-top: 10px;
        ''')
        left_layout.addWidget(self.upload_btn, alignment=Qt.AlignCenter)

        main_layout.addWidget(left_widget)

        # 右侧区域：识别结果详情
        right_widget = QWidget()
        right_widget.setStyleSheet('background-color: white; border-radius: 10px; padding: 20px;')
        right_layout = QVBoxLayout()
        right_widget.setLayout(right_layout)

        # 识别景区按钮
        self.recognize_btn = QPushButton('🔍 识别景区')
        self.recognize_btn.clicked.connect(self.recognize_spot)
        self.recognize_btn.setStyleSheet('''
            background-color: #ff7600;
            color: white;
            border-radius: 8px;
            padding: 12px 20px;
            font-size: 16px;
            font-weight: bold;
            min-width: 300px;
        ''')
        self.recognize_btn.setEnabled(False)  # 初始禁用
        right_layout.addWidget(self.recognize_btn)

        # 示例图片提示
        example_label = QLabel('📷 示例图片 (点击使用)')
        example_label.setStyleSheet('color: #7f8c8d; font-size: 12px; margin: 10px 0;')
        right_layout.addWidget(example_label)

        # 示例图片行
        example_layout = QHBoxLayout()

        # 示例1按钮
        example1_btn = QPushButton('示例1')
        example1_btn.setStyleSheet('''
            background-color: #ecf0f1;
            border-radius: 5px;
            padding: 5px 10px;
            font-size: 12px;
        ''')
        example1_btn.clicked.connect(lambda: self.load_example_image(1))

        # 示例2按钮
        example2_btn = QPushButton('示例2')
        example2_btn.setStyleSheet('''
            background-color: #ecf0f1;
            border-radius: 5px;
            padding: 5px 10px;
            font-size: 12px;
        ''')
        example2_btn.clicked.connect(lambda: self.load_example_image(2))

        example_layout.addWidget(example1_btn)
        example_layout.addWidget(example2_btn)
        right_layout.addLayout(example_layout)

        # 分隔线
        separator = QLabel()
        separator.setStyleSheet('background-color: #ecf0f1; height: 1px; margin: 15px 0;')
        right_layout.addWidget(separator)

        # 景区名称
        spot_name_container = QWidget()
        spot_name_container.setStyleSheet(
            'background-color: #f8f9fa; border-radius: 8px; padding: 10px; margin-bottom: 10px;')
        spot_name_layout = QVBoxLayout()
        spot_name_container.setLayout(spot_name_layout)

        spot_name_title = QLabel('📍 景区名称')
        spot_name_title.setStyleSheet('font-weight: bold; color: #2c3e50; margin-bottom: 5px;')
        spot_name_layout.addWidget(spot_name_title)

        self.spot_name_input = QLineEdit()
        self.spot_name_input.setStyleSheet('''
            background-color: white;
            border: 1px solid #bdc3c7;
            border-radius: 5px;
            padding: 8px;
            font-size: 14px;
        ''')
        self.spot_name_input.setReadOnly(True)
        spot_name_layout.addWidget(self.spot_name_input)

        right_layout.addWidget(spot_name_container)

        # 详细地址
        address_container = QWidget()
        address_container.setStyleSheet(
            'background-color: #f8f9fa; border-radius: 8px; padding: 10px; margin-bottom: 10px;')
        address_layout = QVBoxLayout()
        address_container.setLayout(address_layout)

        address_title = QLabel('🏠 详细地址')
        address_title.setStyleSheet('font-weight: bold; color: #2c3e50; margin-bottom: 5px;')
        address_layout.addWidget(address_title)

        self.address_input = QLineEdit()
        self.address_input.setStyleSheet('''
            background-color: white;
            border: 1px solid #bdc3c7;
            border-radius: 5px;
            padding: 8px;
            font-size: 14px;
        ''')
        self.address_input.setReadOnly(True)
        address_layout.addWidget(self.address_input)

        right_layout.addWidget(address_container)

        # 景点介绍
        intro_container = QWidget()
        intro_container.setStyleSheet(
            'background-color: #f8f9fa; border-radius: 8px; padding: 10px; margin-bottom: 10px;')
        intro_layout = QVBoxLayout()
        intro_container.setLayout(intro_layout)

        intro_title = QLabel('📄 景点介绍')
        intro_title.setStyleSheet('font-weight: bold; color: #2c3e50; margin-bottom: 5px;')
        intro_layout.addWidget(intro_title)

        self.intro_input = QTextEdit()
        self.intro_input.setStyleSheet('''
            background-color: white;
            border: 1px solid #bdc3c7;
            border-radius: 5px;
            padding: 8px;
            font-size: 14px;
        ''')
        self.intro_input.setReadOnly(True)
        intro_layout.addWidget(self.intro_input)

        right_layout.addWidget(intro_container)

        # 门票信息
        ticket_container = QWidget()
        ticket_container.setStyleSheet(
            'background-color: #f8f9fa; border-radius: 8px; padding: 10px; margin-bottom: 10px;')
        ticket_layout = QVBoxLayout()
        ticket_container.setLayout(ticket_layout)

        ticket_title = QLabel('🎫 门票信息')
        ticket_title.setStyleSheet('font-weight: bold; color: #2c3e50; margin-bottom: 5px;')
        ticket_layout.addWidget(ticket_title)

        self.ticket_input = QLineEdit()
        self.ticket_input.setStyleSheet('''
            background-color: white;
            border: 1px solid #bdc3c7;
            border-radius: 5px;
            padding: 8px;
            font-size: 14px;
        ''')
        self.ticket_input.setReadOnly(True)
        ticket_layout.addWidget(self.ticket_input)

        right_layout.addWidget(ticket_container)

        # 开放时间
        open_time_container = QWidget()
        open_time_container.setStyleSheet(
            'background-color: #f8f9fa; border-radius: 8px; padding: 10px; margin-bottom: 10px;')
        open_time_layout = QVBoxLayout()
        open_time_container.setLayout(open_time_layout)

        open_time_title = QLabel('⏰ 开放时间')
        open_time_title.setStyleSheet('font-weight: bold; color: #2c3e50; margin-bottom: 5px;')
        open_time_layout.addWidget(open_time_title)

        self.open_time_input = QLineEdit()
        self.open_time_input.setStyleSheet('''
            background-color: white;
            border: 1px solid #bdc3c7;
            border-radius: 5px;
            padding: 8px;
            font-size: 14px;
        ''')
        self.open_time_input.setReadOnly(True)
        open_time_layout.addWidget(self.open_time_input)

        right_layout.addWidget(open_time_container)

        # 特色景点
        feature_container = QWidget()
        feature_container.setStyleSheet(
            'background-color: #f8f9fa; border-radius: 8px; padding: 10px; margin-bottom: 10px;')
        feature_layout = QVBoxLayout()
        feature_container.setLayout(feature_layout)

        feature_title = QLabel('🌟 特色景点')
        feature_title.setStyleSheet('font-weight: bold; color: #2c3e50; margin-bottom: 5px;')
        feature_layout.addWidget(feature_title)

        self.feature_input = QTextEdit()
        self.feature_input.setStyleSheet('''
            background-color: white;
            border: 1px solid #bdc3c7;
            border-radius: 5px;
            padding: 8px;
            font-size: 14px;
        ''')
        self.feature_input.setReadOnly(True)
        feature_layout.addWidget(self.feature_input)

        right_layout.addWidget(feature_container)

        # 推荐游玩提示
        tips_container = QWidget()
        tips_container.setStyleSheet(
            'background-color: #ebf5fb; border-radius: 8px; padding: 10px; margin-bottom: 10px;')
        tips_layout = QVBoxLayout()
        tips_container.setLayout(tips_layout)

        tips_title = QLabel('🎯 推荐游玩提示')
        tips_title.setStyleSheet('font-weight: bold; color: #2c3e50; margin-bottom: 5px;')
        tips_layout.addWidget(tips_title)

        self.tips_input = QTextEdit()
        self.tips_input.setStyleSheet('''
            background-color: white;
            border: 1px solid #bdc3c7;
            border-radius: 5px;
            padding: 8px;
            font-size: 14px;
        ''')
        self.tips_input.setReadOnly(True)
        tips_layout.addWidget(self.tips_input)

        right_layout.addWidget(tips_container)

        # 实用小贴士
        advice_container = QWidget()
        advice_container.setStyleSheet(
            'background-color: #fef9e7; border-radius: 8px; padding: 10px; margin-bottom: 10px;')
        advice_layout = QVBoxLayout()
        advice_container.setLayout(advice_layout)

        advice_title = QLabel('💡 实用小贴士')
        advice_title.setStyleSheet('font-weight: bold; color: #2c3e50; margin-bottom: 5px;')
        advice_layout.addWidget(advice_title)

        self.advice_input = QTextEdit()
        self.advice_input.setStyleSheet('''
            background-color: white;
            border: 1px solid #bdc3c7;
            border-radius: 5px;
            padding: 8px;
            font-size: 14px;
        ''')
        self.advice_input.setReadOnly(True)
        advice_layout.addWidget(self.advice_input)

        right_layout.addWidget(advice_container)

        main_layout.addWidget(right_widget)

    def load_model(self):
        """加载景点识别模型"""
        try:
            print("🚀 开始加载模型...")
            self.model = load_model()
            print("✅ 模型加载成功")

            # 加载标签
            with open('../spot_labels.txt', 'r', encoding='utf-8') as f:
                self.spot_labels = [line.strip() for line in f.readlines()]
            print(f"✅ 加载了 {len(self.spot_labels)} 个景点标签")

        except Exception as e:
            print(f"❌ 模型加载失败: {str(e)}")
            QMessageBox.critical(self, '错误', f'景点识别模型加载失败: {str(e)}')

    def recognize_spot(self):
        """识别景点"""
        if not self.current_image_path or not self.model or not self.spot_labels:
            QMessageBox.warning(self, '警告', '请先加载图像和识别模型')
            return

        try:
            # 预处理
            input_batch = preprocess_image(self.current_image_path)

            # 预测
            confidence, class_id = get_prediction(self.model, input_batch)

            # 如果有有效结果
            if class_id < len(self.spot_labels):
                spot_name = self.spot_labels[class_id]
                confidence_percent = confidence * 100

                # 更新置信度显示
                self.confidence_label.setText(f'置信度: {confidence_percent:.1f}%')

                # 填充右侧信息框
                self.spot_name_input.setText(spot_name)

                # 根据识别的景点填充详细信息
                spot_info = get_spot_info(spot_name)

                self.address_input.setText(spot_info['address'])
                self.intro_input.setText(spot_info['intro'])
                self.ticket_input.setText(spot_info['ticket'])
                self.open_time_input.setText(spot_info['open_time'])
                self.feature_input.setText(spot_info['features'])
                self.tips_input.setText(spot_info['tips'])
                self.advice_input.setText(spot_info['advice'])

            else:
                QMessageBox.warning(self, '识别失败', '未能识别出有效景点，请尝试其他图片')

        except Exception as e:
            QMessageBox.critical(self, '错误', f'景点识别失败: {str(e)}')

    def upload_image(self):
        """上传图像文件"""
        file_name, _ = QFileDialog.getOpenFileName(
            self, '选择广西景点图像文件', '',
            '图像文件 (*.jpg *.jpeg *.png *.bmp *.gif)'
        )

        if file_name:
            self.current_image_path = file_name

            # 更新按钮状态
            self.upload_btn.setText('✅ 图片已上传')
            self.upload_btn.setStyleSheet('''
                background-color: #2ecc71;
                color: white;
                border-radius: 8px;
                padding: 10px 20px;
                font-weight: bold;
                margin-top: 10px;
            ''')

            self.recognize_btn.setEnabled(True)
            self.clear_result_fields()

    def load_example_image(self, example_num):
        """加载示例图片"""
        # 这里可以添加示例图片路径
        example_paths = {
            1: '../example1.jpg',  # 您需要提供这些示例图片
            2: '../example2.jpg'
        }

        if example_num in example_paths:
            self.current_image_path = example_paths[example_num]
            # 模拟上传成功
            self.upload_btn.setText('✅ 示例图片已加载')
            self.upload_btn.setStyleSheet('''
                background-color: #2ecc71;
                color: white;
                border-radius: 8px;
                padding: 10px 20px;
                font-weight: bold;
                margin-top: 10px;
            ''')
            self.recognize_btn.setEnabled(True)
            self.clear_result_fields()
        else:
            QMessageBox.warning(self, '提示', '示例图片暂未提供')

    def clear_result_fields(self):
        """清空结果字段"""
        self.spot_name_input.clear()
        self.address_input.clear()
        self.intro_input.clear()
        self.ticket_input.clear()
        self.open_time_input.clear()
        self.feature_input.clear()
        self.tips_input.clear()
        self.advice_input.clear()
        self.confidence_label.setText('置信度: 0.0%')

    def closeEvent(self, event):
        """程序关闭事件"""
        reply = QMessageBox.question(self, '确认退出', '确定要退出广西景点识别系统吗？',
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


def main():
    # 强制使用 CPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # 设置编码
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

    # 切换到脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)

    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = SpotRecognitionApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()