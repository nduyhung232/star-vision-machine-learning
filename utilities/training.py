from flask import jsonify
import numpy as np
import matplotlib.pyplot as plt
from stardist import fill_label_holes, gputools_available
from stardist.models import StarDist2D, Config2D
from tifffile import imread
import os
from csbdeep.utils import normalize
from glob import glob
from tqdm import tqdm
import json
from tensorflow.keras.callbacks import Callback

RAW_DATA_FOLDER = 'stardist/raw_data'
MODEL_FOLDER = 'stardist/models'

# Common variables
EPOCHS = 10  # Số epoch huấn luyện
USE_GPU = gputools_available()
RAYS = 32  # Số lượng tia phát ra từ tâm
GRID = (2, 2)  # Kích thước grid
N_CHANNEL = 1

def training(datarawName, epochs, rays):

    if epochs:
        EPOCHS = int(epochs)
    if rays:
        RAYS = int(rays)

    modelName = datarawName + '_' + rays + '_' + epochs

    # Kiểm tra data raw có tồn tại không
    subdirectory_path = os.path.join(RAW_DATA_FOLDER, datarawName)
    if not os.path.isdir(subdirectory_path):
        print(f"Data raw of {datarawName} does not exist")
        return jsonify({'message': "Data raw does not exist"}), 400

    # Đường dẫn tới các thư mục chứa ảnh và mặt nạ
    image_folder = RAW_DATA_FOLDER + '/' + datarawName + '/images'
    mask_folder = RAW_DATA_FOLDER + '/' + datarawName + '/masks'

    # Đọc các đường dẫn tới các tệp ảnh và mặt nạ
    image_files = sorted(glob(os.path.join(image_folder, '*.tif')))
    mask_files = sorted(glob(os.path.join(mask_folder, '*.tif')))

    # Kiểm tra số lượng tệp ảnh và mặt nạ
    assert len(image_files) == len(mask_files), "Số lượng ảnh và mặt nạ không khớp"

    # Đọc dữ liệu
    X = list(map(imread, image_files))
    Y = list(map(imread, mask_files))

    # Kiểm tra kích thước của các ảnh và mặt nạ
    for x, y in zip(X, Y):
        assert x.shape == y.shape, "Ảnh và mặt nạ phải có cùng kích thước"

    # Trục cần được chuẩn hóa độc lập (theo chiều không gian: trục 0 và 1)
    axis_norm = (0, 1)

    # Chuẩn hóa dữ liệu đầu vào (X) bằng cách scale pixel theo các percentiles 1% và 99.8%
    X = [normalize(x, 1, 99.8, axis=axis_norm) for x in X]
    # Điền lỗ hổng (fill holes) trong nhãn (Y) để đảm bảo nhãn không bị gián đoạn
    Y = [fill_label_holes(y) for y in tqdm(Y)]

    # Kiểm tra số lượng dữ liệu, phải có ít nhất 2 mẫu trở lên
    assert len(X) > 1, "not enough training data"

    rng = np.random.RandomState(42)
    # Hoán vị chỉ số của toàn bộ dữ liệu
    ind = rng.permutation(len(X))

    # Chia dữ liệu thành tập train và validation, tỷ lệ validation là 15%
    n_val = max(1, int(round(0.15 * len(ind))))
    ind_train, ind_val = ind[:-n_val], ind[-n_val:]

    # Lấy dữ liệu train và validation theo các chỉ số
    X_val, Y_val = [X[i] for i in ind_val], [Y[i] for i in ind_val]
    X_trn, Y_trn = [X[i] for i in ind_train], [Y[i] for i in ind_train]

    # In số lượng ảnh, số ảnh dùng để train và số ảnh dùng để validation
    print('number of images: %3d' % len(X))
    print('training:       %3d' % len(X_trn))
    print('validation:     %3d' % len(X_val))

    # Giới hạn bộ nhớ GPU nếu sử dụng GPU để tránh xung đột tài nguyên
    if USE_GPU:
        from csbdeep.utils.tf import limit_gpu_memory
        limit_gpu_memory(0.8)  # Giới hạn GPU sử dụng tối đa 80% bộ nhớ

    # Cấu hình mô hình StarDist
    conf = Config2D(
        n_rays=RAYS,
        grid=GRID,
        use_gpu=USE_GPU,
        n_channel_in=N_CHANNEL,
    )
    print(f'- configration - {conf}')
    vars(conf)  # Hiển thị tất cả các thuộc tính cấu hình

    # Khởi tạo mô hình StarDist2D với cấu hình và thư mục lưu trữ
    model = StarDist2D(conf, name=modelName, basedir=MODEL_FOLDER)

    # Huấn luyện mô hình với dữ liệu train và validation, đồng thời sử dụng augmenter
    history = model.train(X_trn, Y_trn, epochs=EPOCHS, validation_data=(X_val, Y_val), augmenter=augmenter)
    # Tối ưu hóa ngưỡng phân đoạn (thresholds) dựa trên tập validation
    model.optimize_thresholds(X_val, Y_val)

    # Lưu log training history
    save_log_training(history, modelName)

    # Vẽ biểu đồ
    loss_during_training(history, modelName)
    iou_during_training(history, modelName)


# Lưu log training
def save_log_training(history, modelName):
    history_file = os.path.join(MODEL_FOLDER, modelName, "training_history.json")
    with open(history_file, 'w') as f:
        json.dump(history.history, f)
    print(f"Training history saved at {history_file}")


# Vẽ biểu đồ loss
def loss_during_training(history, modelName):
    # Đảm bảo thư mục tồn tại
    os.makedirs(MODEL_FOLDER, exist_ok=True)

    # Đường dẫn đầy đủ đến file
    filepath = os.path.join(MODEL_FOLDER, modelName, 'loss_plot.png')

    # Lấy các giá trị loss từ history
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Áp dụng hàm sigmoid vào các giá trị loss
    train_loss_sigmoid = sigmoid(np.array(train_loss))
    val_loss_sigmoid = sigmoid(np.array(val_loss))

    # Tạo dãy số lượng epoch
    epochs_range = range(1, len(train_loss) + 1)

    # Vẽ biểu đồ
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, train_loss_sigmoid, label='Training Loss (Sigmoid)', marker='o')
    plt.plot(epochs_range, val_loss_sigmoid, label='Validation Loss (Sigmoid)', marker='o')
    plt.title('Loss During Training (Sigmoid)')
    plt.xlabel('Epoch')
    plt.ylabel('Sigmoid Loss')
    plt.legend()
    plt.grid()
    plt.tight_layout()

    # Lưu biểu đồ
    plt.savefig(filepath)
    plt.close()  # Đóng biểu đồ để tránh lỗi bộ nhớ

    print(f"Biểu đồ đã được lưu tại: {filepath}")

# Vẽ biểu đồ IOU
def iou_during_training(history, modelName):
    # Đảm bảo thư mục tồn tại
    os.makedirs(MODEL_FOLDER, exist_ok=True)

    # Đường dẫn lưu biểu đồ
    filepath = os.path.join(MODEL_FOLDER, modelName, 'iou_plot.png')

    # Lấy giá trị IOU từ history (giả sử có 'iou' và 'val_iou' trong history)
    train_iou = history.history.get('iou', [])
    val_iou = history.history.get('val_iou', [])

    if not train_iou or not val_iou:
        print("No IOU data found in history.")
        return

    # Dãy số lượng epoch
    epochs_range = range(1, len(train_iou) + 1)

    # Vẽ biểu đồ
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, train_iou, label='Training IOU', marker='o')
    plt.plot(epochs_range, val_iou, label='Validation IOU', marker='o')
    plt.title('IOU During Training')
    plt.xlabel('Epoch')
    plt.ylabel('IOU')
    plt.legend()
    plt.grid()
    plt.tight_layout()

    # Lưu biểu đồ
    plt.savefig(filepath)
    plt.close()

    print(f"Biểu đồ IOU đã được lưu tại: {filepath}")

def random_fliprot(img, mask):
    assert img.ndim >= mask.ndim
    axes = tuple(range(mask.ndim))
    perm = tuple(np.random.permutation(axes))
    img = img.transpose(perm + tuple(range(mask.ndim, img.ndim)))
    mask = mask.transpose(perm)
    for ax in axes:
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=ax)
            mask = np.flip(mask, axis=ax)
    return img, mask

def random_intensity_change(img):
    img = img * np.random.uniform(0.6, 2) + np.random.uniform(-0.2, 0.2)
    return img

def augmenter(x, y):
    """Augmentation of a single input/label image pair.
    x is an input image
    y is the corresponding ground-truth label image
    """
    # Áp dụng phép lật và xoay ngẫu nhiên cho ảnh và nhãn
    x, y = random_fliprot(x, y)

    # Thay đổi cường độ ảnh một cách ngẫu nhiên (tăng/giảm độ sáng)
    x = random_intensity_change(x)

    # Thêm một chút nhiễu Gaussian vào ảnh để tăng độ phong phú cho dữ liệu
    sig = 0.02 * np.random.uniform(0, 1)    # Tạo giá trị sigma ngẫu nhiên từ 0 đến 0.02
    x = x + sig * np.random.normal(0, 1, x.shape)   # Thêm nhiễu Gaussian vào ảnh

    # Trả về ảnh đã được tăng cường (augmented) cùng với nhãn gốc
    return x, y

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class IOUMetric(Callback):
    def __init__(self, val_data):
        self.val_data = val_data  # Validation data (X_val, Y_val)
        self.train_ious = []
        self.val_ious = []

    def on_epoch_end(self, epoch, logs=None):
        # Tính IOU trên tập validation
        X_val, Y_val = self.val_data
        val_pred = self.model.predict(X_val)
        val_iou = self.calculate_iou(Y_val, val_pred)
        self.val_ious.append(val_iou)

        logs["val_iou"] = val_iou  # Ghi log IOU vào history

    @staticmethod
    def calculate_iou(y_true, y_pred):
        # Hàm tính toán IOU cơ bản
        intersection = np.logical_and(y_true, y_pred).sum()
        union = np.logical_or(y_true, y_pred).sum()
        return intersection / union if union != 0 else 0

training(datarawName='bubble-v2', epochs='3', rays='32')