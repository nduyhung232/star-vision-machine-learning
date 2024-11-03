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
import sys
from tqdm import tqdm


stardist_controller = Blueprint('stardist_controller', __name__)

RAW_DATA_FOLDER = 'startdist/raw_data'
MODEL_FOLDER = 'startdist/models'
CORS(stardist_controller, origins=["http://localhost:8080"])

@stardist_controller.route('/api/v1.0/segmentation', methods=['POST'])
def get_image():
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
    model = StarDist2D(None, name=modelName, basedir='models')

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

# Do training
np.random.seed(42)
lbl_cmap = random_label_cmap()
def plot_img_label(img, lbl, img_title="image", lbl_title="label", **kwargs):
    fig, (ai, al) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw=dict(width_ratios=(1.25, 1)))
    im = ai.imshow(img, cmap='gray', clim=(0, 1))
    ai.set_title(img_title)
    fig.colorbar(im, ax=ai)
    al.imshow(lbl, cmap=lbl_cmap)
    al.set_title(lbl_title)
    plt.tight_layout()

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
    x, y = random_fliprot(x, y)
    x = random_intensity_change(x)
    # add some gaussian noise
    sig = 0.02 * np.random.uniform(0, 1)
    x = x + sig * np.random.normal(0, 1, x.shape)
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

    # Tạo cấu hình cho mô hình StarDist
    n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]
    axis_norm = (0, 1)  # normalize channels independently
    print("Normalizing images...")
    if n_channel > 1:
        print(
            "Normalizing image channels %s." % ('jointly' if axis_norm is None or 2 in axis_norm else 'independently'))
        sys.stdout.flush()

    X = [normalize(x, 1, 99.8, axis=axis_norm) for x in X]
    Y = [fill_label_holes(y) for y in tqdm(Y)]

    assert len(X) > 1, "not enough training data"
    rng = np.random.RandomState(42)
    ind = rng.permutation(len(X))
    n_val = max(1, int(round(0.15 * len(ind))))
    ind_train, ind_val = ind[:-n_val], ind[-n_val:]
    X_val, Y_val = [X[i] for i in ind_val], [Y[i] for i in ind_val]
    X_trn, Y_trn = [X[i] for i in ind_train], [Y[i] for i in ind_train]
    print('number of images: %3d' % len(X))
    print('- training:       %3d' % len(X_trn))
    print('- validation:     %3d' % len(X_val))

    i = min(9, len(X) - 1)
    img, lbl = X[i], Y[i]
    assert img.ndim in (2, 3)
    img = img if (img.ndim == 2 or img.shape[-1] == 3) else img[..., 0]
    plot_img_label(img, lbl)
    None;

    print(Config2D.__doc__)

    # 32 is a good default choice (see 1_data.ipynb)
    n_rays = 32

    # Use OpenCL-based computations for data generator during training (requires 'gputools')
    use_gpu = False and gputools_available()

    # Predict on subsampled grid for increased efficiency and larger field of view
    grid = (2, 2)

    conf = Config2D(
        n_rays=n_rays,
        grid=grid,
        use_gpu=use_gpu,
        n_channel_in=n_channel,
    )
    print(conf)
    vars(conf)

    if use_gpu:
        from csbdeep.utils.tf import limit_gpu_memory

        # adjust as necessary: limit GPU memory to be used by TensorFlow to leave some to OpenCL-based computations
        limit_gpu_memory(0.8)
        # alternatively, try this:
        # limit_gpu_memory(None, allow_growth=True)

    model = StarDist2D(conf, name=modelName, basedir='models')

    median_size = calculate_extents(list(Y), np.median)
    fov = np.array(model._axes_tile_overlap('YX'))
    print(f"median object size:      {median_size}")
    print(f"network field of view :  {fov}")
    if any(median_size > fov):
        print("WARNING: median object size larger than field of view of the neural network.")

    # plot some augmented examples
    img, lbl = X[0], Y[0]
    plot_img_label(img, lbl)
    for _ in range(3):
        img_aug, lbl_aug = augmenter(img, lbl)
        plot_img_label(img_aug, lbl_aug, img_title="image augmented", lbl_title="label augmented")


    model.train(X_trn, Y_trn, epochs=10, validation_data=(X_val, Y_val), augmenter=augmenter)
    model.optimize_thresholds(X_val, Y_val)

    return jsonify({'message': 'Training completed!'}), 200