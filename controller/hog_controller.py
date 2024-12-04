import numpy as np
from flask import Blueprint
from flask_cors import CORS
from flask import request, jsonify, url_for
import cv2
import os
from PIL import Image
import uuid
from datetime import datetime

hog = Blueprint('hog', __name__)

HOG_FOLDER = 'hog_images'
CORS(hog, origins=["http://localhost:8080"])


@hog.route('/api/v1.0/hog', methods=['POST'])
def hog_segmentation():
    # Kiểm tra xem có file ảnh trong request không
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    # Đảm bảo thư mục HOG_FOLDER tồn tại
    if not os.path.exists('static/' + HOG_FOLDER):
        os.makedirs('static/' + HOG_FOLDER, exist_ok=True)

    # Lấy file ảnh từ request
    file = request.files['image']
    image = Image.open(file.stream).convert('L')  # Chuyển ảnh thành grayscale

    # Chuyển ảnh thành mảng numpy
    image_np = np.array(image)

    # Áp dụng thuật toán HOG
    cell_size = (8, 8)  # Kích thước ô
    block_size = (2, 2)  # Số ô trong mỗi khối
    bins = 9  # Số hướng
    hog_features, hog_image = hog(image_np,
                                  pixels_per_cell=cell_size,
                                  cells_per_block=block_size,
                                  orientations=bins,
                                  visualize=True,
                                  channel_axis=None)

    # Lưu ảnh HOG vào thư mục static
    file_name = generate_filename()
    hog_image_path = os.path.join('static/' + HOG_FOLDER, file_name)
    cv2.imwrite(hog_image_path, (hog_image * 255).astype('uint8'))  # Chuẩn hóa ảnh HOG

    return jsonify({
        'hog_image_url': url_for('static', filename=f'{HOG_FOLDER}/{file_name}')
    }), 200


def generate_filename(extension='png'):
    timestamp = datetime.now().strftime("%d%m%y-%H%M%S")
    unique_id = uuid.uuid4()  # Tạo UUID ngẫu nhiên
    return f"{timestamp}-{unique_id}.{extension}"
