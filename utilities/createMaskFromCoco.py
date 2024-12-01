import json
import numpy as np
from skimage.draw import polygon
from tifffile import imwrite
import os
from tqdm import tqdm

# Ngưỡng kích thước tối thiểu (width, height)
min_width = 256
min_height = 256

def create_mask_from_coco(json_file, output_mask_folder):
    # Tải dữ liệu COCO
    with open(json_file, 'r') as f:
        coco_data = json.load(f)

    # Tạo thư mục lưu mask nếu chưa tồn tại
    os.makedirs(output_mask_folder, exist_ok=True)

    # Duyệt qua từng ảnh trong dữ liệu
    for img_info in tqdm(coco_data['images']):
        # Thông tin ảnh
        img_id = img_info['id']
        width = img_info['width']
        height = img_info['height']
        file_name = img_info['file_name']

        # Tạo mask đa lớp (ban đầu tất cả là 0 - nền)
        mask = np.zeros((height, width), dtype=np.uint8)

        # Lấy annotations liên quan đến ảnh này
        annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == img_id]

        # Duyệt qua từng annotation
        for ann in annotations:
            category_id = ann['category_id']  # Lớp của đối tượng
            for seg in ann['segmentation']:
                # Xây dựng polygon từ segmentation
                poly = np.array(seg).reshape(-1, 2)
                rr, cc = polygon(poly[:, 1], poly[:, 0], mask.shape)
                mask[rr, cc] = category_id  # Gán giá trị lớp

        # Lưu mask thành tệp ảnh
        mask_file_name = os.path.splitext(file_name)[0] + '_mask.tif'
        mask_path = os.path.join(output_mask_folder, mask_file_name)
        imwrite(mask_path, mask)


# Thông tin đầu vào
json_file = 'data/bubble-dataset/valid/_annotations.coco.json'
images_folder = 'data/bubble-dataset/valid/origin'
output_mask_folder = 'data/bubble-dataset/valid/masks'
output_format = 'tif'

# Tạo mask
create_mask_from_coco(json_file, images_folder, output_mask_folder)
