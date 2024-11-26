import os
from glob import glob
from skimage.io import imread, imsave
import numpy as np

# Đường dẫn đến thư mục chứa ảnh gốc và mask
image_folder = 'E:\star-vision\star-vision-machine-learning\Class1\images'
mask_folder = 'E:\star-vision\star-vision-machine-learning\Class1\masks'
output_folder = 'E:\star-vision\star-vision-machine-learning\Class1\output'

# Tạo thư mục output nếu chưa tồn tại
os.makedirs(output_folder, exist_ok=True)

# Lấy danh sách ảnh gốc
image_files = sorted(glob(os.path.join(image_folder, '*.jpg')))  # Thay đổi định dạng ảnh gốc nếu cần

# Xử lý từng ảnh
for image_file in image_files:
    # Lấy tên ảnh (không bao gồm phần mở rộng)
    image_name = os.path.splitext(os.path.basename(image_file))[0]

    # Tìm các mask tương ứng với ảnh này
    mask_files = sorted(glob(os.path.join(mask_folder, f"{image_name}_*.jpg")))  # Đọc các mask theo pattern

    if not mask_files:
        print(f"Không tìm thấy mask cho ảnh {image_name}")
        continue

    # Đọc và gộp các mask
    merged_mask = None
    for mask_file in mask_files:
        mask = imread(mask_file)
        if merged_mask is None:
            merged_mask = mask
        else:
            merged_mask = np.maximum(merged_mask, mask)  # Sử dụng giá trị lớn nhất (nếu mask chồng lên nhau)

    # Lưu mask gộp thành file .tif
    output_path = os.path.join(output_folder, f"{image_name}.tif")
    imsave(output_path, merged_mask.astype(np.uint8))  # Lưu dưới dạng uint8
    print(f"Đã lưu mask gộp cho {image_name} tại {output_path}")
