import os
from PIL import Image

# Đường dẫn tới folder chứa ảnh
folder_path = "C:/Users/ADMIN/Downloads/BubANN_GAN.v1i.coco-segmentation/train"

# Ngưỡng kích thước tối thiểu (width, height)
min_width = 640
min_height = 640

# Duyệt qua tất cả file trong folder
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)

    # Kiểm tra nếu file là ảnh
    try:
        with Image.open(file_path) as img:
            width, height = img.size

        # Xóa ảnh nếu kích thước nhỏ hơn ngưỡng
        if width < min_width or height < min_height:
            print(f"Deleting {filename}: {width}x{height}")
            os.remove(file_path)
    except Exception as e:
        print(f"Skipping {filename}: {e}")
