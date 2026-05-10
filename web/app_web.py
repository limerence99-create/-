import os
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
from core.model import load_model
from core.utils import preprocess_image, get_prediction, get_spot_info

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 加载模型
model = load_model()

# 修复 spot_labels.txt 路径
import pathlib
project_root = pathlib.Path(__file__).resolve().parent.parent
spot_labels_path = project_root / 'spot_labels.txt'

try:
    with open(spot_labels_path, 'r', encoding='utf-8') as f:
        spot_labels = [line.strip() for line in f.readlines()]
except Exception as e:
    print(f"⚠️ 无法加载 spot_labels.txt: {e}")
    spot_labels = ['桂林山水', '阳朔西街', '漓江']  # 默认兜底数据


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': '没有文件'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'})

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # 预处理
        input_batch = preprocess_image(filepath)

        # 预测
        confidence, class_id = get_prediction(model, input_batch)

        if class_id < len(spot_labels):
            spot_name = spot_labels[class_id]
            confidence_percent = confidence * 100

            # 获取景点信息
            spot_info = get_spot_info(spot_name)

            # 记录历史（简单内存记录）
            history_entry = {
                'filename': filename,
                'spot_name': spot_name,
                'confidence': confidence_percent,
                'timestamp': str(datetime.now())[:19]  # 只保留到秒
            }
            # 简单存储在全局变量中（生产环境建议用数据库或文件）
            if not hasattr(app, 'history'):
                app.history = []
            app.history.append(history_entry)

            return jsonify({
                'success': True,
                'spot_name': spot_name,
                'confidence': confidence_percent,
                'address': spot_info['address'],
                'intro': spot_info['intro'],
                'ticket': spot_info['ticket'],
                'open_time': spot_info['open_time'],
                'features': spot_info['features'],
                'tips': spot_info['tips'],
                'advice': spot_info['advice']
            })
        else:
            return jsonify({'error': '未能识别出有效景点'})

    except Exception as e:
        print(f"❌ 识别过程发生错误: {str(e)}")
        return jsonify({'error': f'上传或识别过程中发生错误: {str(e)}'})


@app.route('/history')
def get_history():
    history_data = getattr(app, 'history', [])
    return jsonify({'history': history_data})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)