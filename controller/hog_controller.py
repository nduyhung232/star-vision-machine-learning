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
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']
    image = Image.open(file.stream).convert('RGB')
    image_np = np.array(image)

    # 1. Khởi tạo tham số cho HOG
    cell_size = (8, 8)
    block_size = (2, 2)
    bins = 9  # Số hướng của gradient

    # 2. Tính toán đặc trưng HOG và trực quan hóa HOG (chỉ rõ channel_axis)
    hog_features, hog_image = hog(image_np,
                                  pixels_per_cell=cell_size,
                                  cells_per_block=block_size,
                                  orientations=bins,
                                  visualize=True,
                                  channel_axis=-1)  # Chỉ rõ trục màu cuối cùng

    # 3. Tạo ảnh HOG để trực quan hóa và lưu lại
    hog_image_file_name = generate_filename()
    hog_image_path = os.path.join(f'static/{HOG_FOLDER}', hog_image_file_name)
    cv2.imwrite(hog_image_path, (hog_image * 255).astype('uint8'))

    # 4. Áp dụng GrabCut để phân đoạn ảnh
    mask = np.zeros(image_np.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    rect = (10, 10, image_np.shape[1] - 10, image_np.shape[0] - 10)
    cv2.grabCut(image_np, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    segmented_image = image_np * mask2[:, :, np.newaxis]

    # 5. Tìm các bounding boxes
    gray_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append({'x': x, 'y': y, 'width': w, 'height': h})

    segmented_image_with_boxes = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)
    for box in bounding_boxes:
        x, y, w, h = box['x'], box['y'], box['width'], box['height']
        cv2.rectangle(segmented_image_with_boxes, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # 6. Lưu ảnh phân đoạn
    segmented_image_file_name = generate_filename()
    segmented_image_path = os.path.join(f'static/{HOG_FOLDER}', segmented_image_file_name)
    cv2.imwrite(segmented_image_path, cv2.cvtColor(segmented_image_with_boxes, cv2.COLOR_RGB2BGR))

    return jsonify({
        'hog_image_url': url_for('static', filename=f'{HOG_FOLDER}/{hog_image_file_name}'),
        'segmented_image_url': url_for('static', filename=f'{HOG_FOLDER}/{segmented_image_file_name}'),
        'bounding_boxes': bounding_boxes,
        'total_bounding_boxes': len(bounding_boxes)
    }), 200


def generate_filename(extension='png'):
    timestamp = datetime.now().strftime("%d%m%y-%H%M%S")
    unique_id = uuid.uuid4()  # Tạo UUID ngẫu nhiên
    return f"{timestamp}-{unique_id}.{extension}"
