from flask import request, jsonify
import os
import cv2
from skimage.feature import hog
from skimage import exposure
import numpy as np
from flask_cors import CORS
from flask import Blueprint
import uuid
from datetime import datetime

edge = Blueprint('edge', __name__)

EDGE_FOLDER = 'edge_segmentation_images'
CORS(edge, origins=["http://localhost:8080"])


@edge.route('/api/v1.0/hog', methods=['POST'])
def hog_segmentation():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Đọc ảnh và xử lý HOG
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return jsonify({'error': 'Invalid image file'}), 400

    # Tính toán HOG
    fd, hog_image = hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=True,
        channel_axis=None
    )

    # Tăng cường độ tương phản để hiển thị kết quả HOG
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    # Lưu ảnh kết quả
    output_filename = f"hog_{file.filename}"
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
    cv2.imwrite(output_path, (hog_image_rescaled * 255).astype(np.uint8))

    return jsonify({'message': 'Image processed successfully', 'output_path': output_path}), 200


def generate_filename(extension='png'):
    timestamp = datetime.now().strftime("%d%m%y-%H%M%S")
    unique_id = uuid.uuid4()  # Tạo UUID ngẫu nhiên
    return f"{timestamp}-{unique_id}.{extension}"
