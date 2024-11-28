from flask import Blueprint
from flask import Response, request, jsonify
from io import BytesIO
from flask_cors import CORS
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from stardist import random_label_cmap, fill_label_holes, calculate_extents, gputools_available
from stardist.models import StarDist2D, Config2D
from tifffile import imread
import base64
import os
from csbdeep.utils import normalize
from glob import glob
from tqdm import tqdm
import json

stardist_controller = Blueprint('stardist_controller', __name__)

RAW_DATA_FOLDER = 'stardist/raw_data'
MODEL_FOLDER = 'stardist/models'
CORS(stardist_controller, origins=["http://localhost:8080"])


@stardist_controller.route('/api/v1.0/segmentation', methods=['POST'])
def segmentation():
    # Open image from request:
    imageUpload = request.files['image']
    modelName = request.form['modelName']
    modelColorMap = request.form['colorMap']

    # Open image by pilow and normalize image
    axis_norm = (0, 1)
    image = Image.open(imageUpload)
    image = image.convert('L')
    image = np.array(image)
    image = normalize(image, 1, 99.8, axis=axis_norm)

    # Open trained model
    model = StarDist2D(None, name=modelName, basedir=MODEL_FOLDER)

    # Predict the image
    y_test = model.predict_instances(image, n_tiles=model._guess_n_tiles(image), show_tile_progress=False)

    # Define a colormap
    cmap = random_label_cmap()
    if modelColorMap != "random_label_cmap":
        cmap = plt.get_cmap(modelColorMap)

    # Normalize the grayscale image values to the range [0, 1]
    normalized_image = (y_test[0] - y_test[0].min()) / (y_test[0].max() - y_test[0].min())

    # Apply the colormap to the normalized image
    image = (cmap(normalized_image) * 255).astype(np.uint8)
    image = Image.fromarray(image)
    # image.save('output.png')

    # Save the image to a BytesIO object
    image_bytes = BytesIO()
    image.save(image_bytes, format='PNG')
    image_bytes.seek(0)

    # Return the image as a response with the appropriate content type
    # return Response(image_bytes, content_type='image/png')
    image_base64 = base64.b64encode(image_bytes.getvalue()).decode('utf-8')

    # Create a dictionary to hold the image and text data
    response_data = {
        'imageBytes': image_base64,
        'objectsCount': len(y_test[1]['points']),
        'points': y_test[1]['points'].tolist(),
        'coord': y_test[1]['coord'].tolist(),
    }

    # Return the response as JSON with image and text data
    return jsonify(response_data)


@stardist_controller.route('/api/v1.0/convert', methods=['POST'])
def convert_tiff_to_png():
    tiff_file = request.files['tiff_file']
    tiff_image = Image.open(tiff_file)
    if tiff_file and (tiff_file.filename.endswith('.tif') or tiff_file.filename.endswith('.tiff')):
        tiff_image = tiff_image.convert('RGB')
    output_buffer = BytesIO()
    tiff_image.save(output_buffer, format='PNG')
    output_buffer.seek(0)
    return Response(output_buffer, content_type='image/png')


### Do training
# Common variables
EPOCHS = 10  # Số epoch huấn luyện
USE_GPU = gputools_available()
RAYS = 32  # Số lượng tia phát ra từ tâm
GRID = (2, 2)  # Kích thước grid
N_CHANNEL = 1

rng = np.random.seed(42)
lbl_cmap = random_label_cmap()


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


@stardist_controller.route('/api/v1.0/training', methods=['POST'])
def training():
    modelName = request.form['modelName']

    # Kiểm tra model có tồn tại không
    subdirectory_path = os.path.join(RAW_DATA_FOLDER, modelName)
    if not os.path.isdir(subdirectory_path):
        print(f"Data raw of {modelName} does not exist")
        return jsonify({'message': "Data raw does not exist"}), 400

    # Đường dẫn tới các thư mục chứa ảnh và mặt nạ
    image_folder = RAW_DATA_FOLDER + '/' + modelName + '/images'
    mask_folder = RAW_DATA_FOLDER + '/' + modelName + '/masks'

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

    # Giới hạn bộ nhớ GPU nếu sử dụng GPU để tránh xung đột tài nguyên
    if USE_GPU:
        from csbdeep.utils.tf import limit_gpu_memory
        limit_gpu_memory(0.8)  # Giới hạn GPU sử dụng tối đa 80% bộ nhớ

    # Trục cần được chuẩn hóa độc lập (theo chiều không gian: trục 0 và 1)
    axis_norm = (0, 1)

    # Chuẩn hóa dữ liệu đầu vào (X) bằng cách scale pixel theo các percentiles 1% và 99.8%
    X = [normalize(x, 1, 99.8, axis=axis_norm) for x in X]
    # Điền lỗ hổng (fill holes) trong nhãn (Y) để đảm bảo nhãn không bị gián đoạn
    Y = [fill_label_holes(y) for y in tqdm(Y)]

    # Kiểm tra số lượng dữ liệu, phải có ít nhất 2 mẫu
    assert len(X) > 1, "not enough training data"

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
    print('- training:       %3d' % len(X_trn))
    print('- validation:     %3d' % len(X_val))

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

    return jsonify({'message': 'Training completed!'}), 200


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

    # Vẽ và lưu biểu đồ
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(1, len(train_loss) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, train_loss, label='Training Loss', marker='o')
    plt.plot(epochs_range, val_loss, label='Validation Loss', marker='o')
    plt.title('Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(filepath)
    plt.close()  # Đóng biểu đồ để tránh lỗi bộ nhớ

    print(f"Biểu đồ đã được lưu tại: {filepath}")
