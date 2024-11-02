from flask import Blueprint
from werkzeug.utils import secure_filename
import os
import zipfile
import shutil
from flask_cors import CORS
from flask import request, jsonify

file_management = Blueprint('file_management', __name__)

RAW_DATA_FOLDER = 'raw_data'
MODEL_FOLDER = 'models'
CORS(file_management, origins=["http://localhost:8080"])


# Get list model
@file_management.route('/api/v1.0/models', methods=['GET'])
def get_models():
    subdir = [name for name in os.listdir(MODEL_FOLDER)]
    return subdir


# Create model: create folder that store model
@file_management.route('/api/v1.0/model', methods=['POST'])
def create_model():
    fileName = request.form['fileName']
    folderPath = RAW_DATA_FOLDER + "/" + fileName
    if os.path.exists(folderPath):
        return jsonify({"message": "File exist"}), 400
    else:
        os.mkdir(folderPath)
        return jsonify({"message": "Success"}), 200


# Delete model: delete folder that store model
@file_management.route('/api/v1.0/model', methods=['DELETE'])
def delete_models():
    modelName = request.args.get('modelName')
    # Delete raw_data
    folderPath = RAW_DATA_FOLDER + "/" + modelName
    if os.path.exists(folderPath):
        try:
            shutil.rmtree(folderPath)
            print(f'Successfully deleted the directory: {folderPath}')
        except FileNotFoundError:
            print(f'The directory {folderPath} does not exist')
        except PermissionError:
            print(f'Permission denied: {folderPath}')
        except Exception as e:
            print(f'Error occurred: {e}')
    else:
        return jsonify({"message": "File raw data not exist"}), 400

    # Delete model
    folderModelPath = MODEL_FOLDER + "/" + modelName
    if os.path.exists(folderModelPath):
        try:
            shutil.rmtree(folderModelPath)
            print(f'Successfully deleted the directory: {folderModelPath}')
        except FileNotFoundError:
            print(f'The directory {folderModelPath} does not exist')
        except PermissionError:
            print(f'Permission denied: {folderModelPath}')
        except Exception as e:
            print(f'Error occurred: {e}')
    else:
        return jsonify({"message": "File model not exist"}), 400

    return jsonify({"message": "Success"}), 200


def check_zip_structure(file_path):
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_structure = zip_ref.namelist()
        return zip_structure


# Upload origins and masks images
@file_management.route('/api/v1.0/upload', methods=['POST'])
def upload_file():
    global file_path
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    modelName = request.form['modelName']

    if not modelName:
        return jsonify({'error': 'No model name'}), 400

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    rawDataModelPath = RAW_DATA_FOLDER + '/' + modelName
    # Check if the folder for storing uploaded files exists, it not create it
    if not os.path.exists(rawDataModelPath):
        os.makedirs(rawDataModelPath)

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(rawDataModelPath, filename)
        file.save(file_path)

        # Kiểm tra cấu trúc file zip
        if zipfile.is_zipfile(file_path):
            zip_structure = check_zip_structure(file_path)
            if zip_structure.__contains__('images/') and zip_structure.__contains__('masks/'):
                # Giải nén file nếu cấu trúc hợp lệ
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(rawDataModelPath)

                result = jsonify({'message': 'File successfully uploaded and extracted'}), 200
            else:
                result = jsonify({'message': 'Uploaded should contains \'masks\' folder and \'images\' folder'}), 400
        else:
            os.remove(file_path)  # Xóa file nếu không phải là file zip hợp lệ
            result = jsonify({'error': 'Uploaded file is not a valid zip file'}), 400
    else:
        result = jsonify({'error': 'No file part in the request'}), 400

    os.remove(file_path)  # Xóa file zip trước khi kết thúc xử lý
    return result
