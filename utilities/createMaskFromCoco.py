import os
import json
from PIL import Image, ImageDraw

# Ngưỡng kích thước tối thiểu (width, height)
min_width = 256
min_height = 256

def create_mask_from_coco(json_path, images_folder, masks_folder, output_format='tif'):
    # Tạo thư mục masks nếu chưa tồn tại
    os.makedirs(masks_folder, exist_ok=True)

    # Load file JSON
    with open(json_path, "r") as file:
        coco_data = json.load(file)

    # Lấy danh sách ảnh và annotations
    images = {img["id"]: img["file_name"] for img in coco_data["images"]}
    annotations = coco_data["annotations"]

    # Duyệt qua từng ảnh
    for image_id, image_name in images.items():
        # Lấy tất cả annotations có image_id trùng khớp
        image_annotations = [ann for ann in annotations if ann["image_id"] == image_id]

        # Load ảnh để biết kích thước (nếu cần)
        image_path = os.path.join(images_folder, image_name)
        if not os.path.exists(image_path):
            continue

        with Image.open(image_path) as img:
            width, height = img.size

        # Khởi tạo mask trống
        mask = Image.new("L", (width, height), 0)  # Mask nhị phân, giá trị 0 hoặc 255
        draw = ImageDraw.Draw(mask)

        # Duyệt qua từng vật thể và vẽ mask
        for ann in image_annotations:
            if "segmentation" in ann and isinstance(ann["segmentation"], list):
                for seg in ann["segmentation"]:
                    polygon = [(seg[i], seg[i + 1]) for i in range(0, len(seg), 2)]
                    draw.polygon(polygon, fill=255)  # Vẽ vật thể với giá trị 255

        # Lưu mask với phần mở rộng .tif
        mask_path = os.path.join(masks_folder, f"{os.path.splitext(image_name)[0]}.tif")
        mask.save(mask_path, format="TIFF")  # Sử dụng "TIFF" cho định dạng file
        print(f"Saved mask for {image_name} to {mask_path}")

# Thông tin đầu vào
json_path = 'C:/Users/Tom Nguyen/Downloads/bubble-dataset/valid/_annotations.coco.json'
images_folder = 'C:/Users/Tom Nguyen/Downloads/bubble-dataset/valid/origin'
masks_folder = 'C:/Users/Tom Nguyen/Downloads/bubble-dataset/valid/masks'
output_format = 'tif'

create_mask_from_coco(json_path, images_folder, masks_folder, output_format)
