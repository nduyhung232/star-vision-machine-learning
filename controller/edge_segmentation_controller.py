import numpy as np
from flask import Blueprint
from flask_cors import CORS
from flask import request, jsonify, url_for
import cv2
import os
from PIL import Image
import uuid
from datetime import datetime

edge = Blueprint('edge', __name__)

EDGE_FOLDER = 'edge_segmentation_images'
CORS(edge, origins=["http://localhost:8080"])


@edge.route('/api/v1.0/canny', methods=['POST'])
def canny_edge_segmentation():
    # Kiểm tra xem có file ảnh trong request không
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    # Đảm bảo thư mục THRESHOLD_FOLDER tồn tại
    if not os.path.exists('static/' + EDGE_FOLDER):
        os.makedirs('static/' + EDGE_FOLDER, exist_ok=True)

    # Lấy file ảnh từ request
    file = request.files['image']
    image = Image.open(file.stream).convert('L')  # Chuyển ảnh thành grayscale

    # Chuyển ảnh thành mảng numpy
    image_np = np.array(image)

    # Áp dụng thuật toán Canny với ngưỡng tùy chọn
    threshold_value1, threshold_value2 = 100, 200  # Các ngưỡng tùy chọn
    edges = cv2.Canny(image_np, threshold1=threshold_value1, threshold2=threshold_value2)

    # Lưu ảnh đã xử lý vào thư mục static
    file_name = generate_filename()
    thresholded_image_path = os.path.join('static/' + EDGE_FOLDER, file_name)
    cv2.imwrite(thresholded_image_path, edges)

    return jsonify({
        'canny_image_url': url_for('static', filename=f'{EDGE_FOLDER}/{file_name}')
    }), 200


def generate_filename(extension='png'):
    timestamp = datetime.now().strftime("%d%m%y-%H%M%S")
    unique_id = uuid.uuid4()  # Tạo UUID ngẫu nhiên
    return f"{timestamp}-{unique_id}.{extension}"
