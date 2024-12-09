from PIL import Image
import os

def convert_jpg_to_tif(input_folder, output_folder):
    # Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.jpg'):  # Chỉ xử lý file .jpg
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.tif')

            try:
                # Mở ảnh và đảm bảo không thay đổi kích thước
                with Image.open(input_path) as img:
                    # Chuyển ảnh sang chế độ Grayscale
                    img = img.convert("L")  # Chế độ "L" cho ảnh grayscale

                    # Lưu ảnh dưới dạng .tif với kích thước không thay đổi
                    img.save(output_path, format='TIFF')

                print(f"Converted: {input_path} -> {output_path}")

            except Exception as e:
                print(f"Error processing {input_path}: {e}")


# Thư mục chứa file .jpg và nơi lưu file .tif
input_folder = "data/BubANN_GAN.v1i.coco-segmentation/test/origin"
output_folder = "data/BubANN_GAN.v1i.coco-segmentation/test/images"

# Gọi hàm chuyển đổi
convert_jpg_to_tif(input_folder, output_folder)
