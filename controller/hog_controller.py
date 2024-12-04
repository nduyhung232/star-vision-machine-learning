import numpy as np
from flask import Blueprint
from flask_cors import CORS
from flask import request, jsonify, url_for
import cv2
import os
from PIL import Image
import uuid
from datetime import datetime
from skimage.feature import hog

hog_bp  = Blueprint('hog_bp ', __name__)

HOG_FOLDER = 'hog_images'
CORS(hog_bp , origins=["http://localhost:8080"])


@hog_bp .route('/api/v1.0/hog', methods=['POST'])
def hog_segmentation():
    # Kiểm tra xem có file ảnh trong request không
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    # Đảm bảo thư mục EDGE_FOLDER tồn tại
    if not os.path.exists('static/' + HOG_FOLDER):
        os.makedirs('static/' + HOG_FOLDER, exist_ok=True)

    # Lấy file ảnh từ request
    file = request.files['image']
    image = Image.open(file.stream).convert('L')  # Chuyển ảnh thành grayscale
    image_np = np.array(image)

    # Áp dụng thuật toán HOG
    cell_size = (8, 8)
    block_size = (2, 2)
    bins = 9
    hog_features, hog_image = hog(image_np,
                                  pixels_per_cell=cell_size,
                                  cells_per_block=block_size,
                                  orientations=bins,
                                  visualize=True,
                                  channel_axis=None)

    # Tạo bounding box giả lập dựa trên gradient
    contours, _ = cv2.findContours((hog_image * 255).astype('uint8'),
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = []
    segmented_image = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append({'x': x, 'y': y, 'width': w, 'height': h})
        cv2.rectangle(segmented_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Lưu ảnh HOG trực quan hóa
    hog_image_file_name = generate_filename()
    hog_image_path = os.path.join('static/' + HOG_FOLDER, hog_image_file_name)
    cv2.imwrite(hog_image_path, (hog_image * 255).astype('uint8'))

    # Trả về JSON response
    return jsonify({
        'hog_image_url': url_for('static', filename=f'{HOG_FOLDER}/{hog_image_file_name}'),
        'bounding_boxes': bounding_boxes,
        'total_bounding_boxes': len(bounding_boxes)
    }), 200


def generate_filename(extension='png'):
    timestamp = datetime.now().strftime("%d%m%y-%H%M%S")
    unique_id = uuid.uuid4()  # Tạo UUID ngẫu nhiên
    return f"{timestamp}-{unique_id}.{extension}"
