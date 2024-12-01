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

    # Đảm bảo thư mục EDGE_FOLDER tồn tại
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

    # Tìm contours từ ảnh Canny
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Tạo danh sách bounding boxes
    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)  # Lấy bounding box
        bounding_boxes.append({'x': x, 'y': y, 'width': w, 'height': h})

    # Vẽ bounding boxes lên ảnh
    segmented_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)  # Chuyển ảnh về màu để vẽ
    for box in bounding_boxes:
        x, y, w, h = box['x'], box['y'], box['width'], box['height']
        cv2.rectangle(segmented_image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Vẽ bounding box màu đỏ

    # Lưu ảnh Canny
    canny_image_file_name = generate_filename()
    canny_image_path = os.path.join('static/' + EDGE_FOLDER, canny_image_file_name)
    cv2.imwrite(canny_image_path, edges)

    # Lưu ảnh đã phân đoạn
    segmented_image_file_name = generate_filename()
    segmented_image_path = os.path.join('static/' + EDGE_FOLDER, segmented_image_file_name)
    cv2.imwrite(segmented_image_path, segmented_image)

    # Trả về JSON với thông tin cần thiết
    return jsonify({
        'edged_image_url': url_for('static', filename=f'{EDGE_FOLDER}/{canny_image_file_name}'),
        'segmented_image_url': url_for('static', filename=f'{EDGE_FOLDER}/{segmented_image_file_name}'),
        'bounding_boxes': bounding_boxes,
        'total_bounding_boxes': len(bounding_boxes)
    }), 200


def generate_filename(extension='png'):
    timestamp = datetime.now().strftime("%d%m%y-%H%M%S")
    unique_id = uuid.uuid4()  # Tạo UUID ngẫu nhiên
    return f"{timestamp}-{unique_id}.{extension}"
