import numpy as np
from flask import Blueprint
from flask_cors import CORS
from flask import request, jsonify, url_for
import cv2
import matplotlib.pyplot as plt
import os

threshold = Blueprint('threshold', __name__)

THRESHOLD_FOLDER = 'threshold_images'
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

    # Đảm bảo thư mục THRESHOLD_FOLDER tồn tại
    if not os.path.exists('static/' + THRESHOLD_FOLDER):
        os.makedirs('static/' + THRESHOLD_FOLDER, exist_ok=True)

    # Áp dụng phương pháp Otsu để phân ngưỡng
    threshold_value, thresholded_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Lưu hình ảnh đã phân ngưỡng
    thresholded_image_path = os.path.join('static/' + THRESHOLD_FOLDER, 'thresholded_image.png')
    cv2.imwrite(thresholded_image_path, thresholded_image)

    # Tính histogram của hình ảnh đầu vào
    histogram, bin_edges = np.histogram(image, bins=256, range=(0, 256))

    # Vẽ histogram và lưu dưới dạng ảnh
    plt.figure(figsize=(10, 5))
    plt.plot(bin_edges[0:-1], histogram, color="black")
    plt.axvline(threshold_value, color='red', linestyle='dashed', linewidth=1.5, label=f'Threshold: {threshold_value}')
    plt.legend()
    plt.title("Histogram with Otsu's Threshold")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")

    # Lưu histogram vào tệp
    histogram_image_path = os.path.join('static/' + THRESHOLD_FOLDER, 'histogram.png')
    plt.savefig(histogram_image_path)
    plt.close()

    # Trả về đường dẫn URL của hình ảnh đã phân ngưỡng và histogram
    return jsonify({
        'thresholded_image_url': url_for('static', filename=f'{THRESHOLD_FOLDER}/thresholded_image.png'),
        'histogram_url': url_for('static', filename=f'{THRESHOLD_FOLDER}/histogram.png'),
        'threshold_value': threshold_value
    })
