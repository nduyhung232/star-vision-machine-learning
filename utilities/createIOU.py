import os
import numpy as np
from stardist.models import StarDist2D
from tifffile import imread
from sklearn.metrics import jaccard_score
import matplotlib.pyplot as plt
from csbdeep.utils import normalize

save_path = '/stardist/models/nucle'

# Đường dẫn tới tập dữ liệu kiểm tra
test_images_folder = "data/nucle-data/test/images"
test_masks_folder = "data/nucle-data/test/masks"

# Nạp mô hình đã được huấn luyện
model_path = "stardist/models/nucle"

def calculate_iou(y_true, y_pred):
    """Hàm tính toán IOU."""
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    return jaccard_score(y_true_flat, y_pred_flat, average='micro')

model = StarDist2D(None, name=model_path)

# Đọc danh sách tệp
image_files = sorted(os.listdir(test_images_folder))
mask_files = sorted(os.listdir(test_masks_folder))

assert len(image_files) == len(mask_files), "Số lượng ảnh và nhãn không khớp."

# Tính toán IOU cho từng ảnh
ious = []
for img_file, mask_file in zip(image_files, mask_files):
    img_path = os.path.join(test_images_folder, img_file)
    mask_path = os.path.join(test_masks_folder, mask_file)

    # Đọc ảnh và nhãn
    img = imread(img_path)
    true_mask = imread(mask_path)

    # Kiểm tra xem ảnh có phải là ảnh 1D hay 2D
    if img.ndim == 1:
        axis_norm = 0  # Dùng axis=0 cho mảng 1D
    elif img.ndim == 2:
        axis_norm = (0, 1)  # Dùng cả hai trục cho mảng 2D
    else:
        axis_norm = (0, 1)  # Mặc định cho các mảng nhiều chiều

    # Chuẩn hóa ảnh
    img = normalize(img, 1, 99.8, axis=axis_norm)

    # Dự đoán bằng mô hình
    pred_mask, _ = model.predict_instances(img)

    # Tính IOU
    iou = calculate_iou(true_mask, pred_mask)
    ious.append(iou)
    print(f"{img_file} IOU: {iou:.4f}")

# Tính IOU trung bình
mean_iou = np.mean(ious)
print(f"Mean IOU: {mean_iou:.4f}")

# Vẽ biểu đồ IOU
image_indices = range(1, len(ious) + 1)

plt.figure(figsize=(10, 5))
plt.bar(range(len(ious)), ious, color='blue', alpha=0.7)
plt.axhline(y=mean_iou, color='r', linestyle='--', label=f'Mean IOU: {mean_iou:.4f}')
plt.xlabel('Test Image Index')
plt.ylabel('IOU')
plt.title('IOU for Test Images')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Lưu biểu đồ vào file PNG trong thư mục đã chỉ định
file_path = os.path.join(save_path, 'iou_line_chart.png')
plt.savefig(file_path)

# Thông báo lưu thành công
print(f"Biểu đồ đã được lưu tại: {file_path}")