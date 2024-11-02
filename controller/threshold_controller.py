import numpy as np
from flask import Blueprint
from flask_cors import CORS
from flask import request, jsonify, send_file
import cv2

threshold = Blueprint('threshold', __name__)

RAW_DATA_FOLDER = 'raw_data'
MODEL_FOLDER = 'models'
CORS(threshold, origins=["http://localhost:8080"])


# Get list model
@threshold.route('/threshold', methods=['POST'])
def threshold_image():
    # Kiểm tra xem có tệp được tải lên không
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']

    # Kiểm tra xem tệp có phải là hình ảnh không
    if not file or not file.filename:
        return jsonify({'error': 'No image file provided'}), 400

    # Đọc hình ảnh từ tệp
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)

    if image is None:
        return jsonify({'error': 'Could not read the image'}), 400

    # Áp dụng phương pháp Otsu để phân ngưỡng
    _, thresholded_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Lưu hình ảnh đã phân ngưỡng
    output_path = 'thresholded_image.png'
    cv2.imwrite(output_path, thresholded_image)

    # Gửi hình ảnh đã phân ngưỡng về client
    return send_file(output_path, mimetype='image/png')