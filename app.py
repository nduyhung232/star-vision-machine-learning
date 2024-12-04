from flask import Flask
from controller.file_management_controller import file_management
from controller.hog_controller import hog
from controller.stardist_controller import stardist_controller
from controller.threshold_controller import threshold
from controller.edge_segmentation_controller import edge
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

app = Flask(__name__)

app.register_blueprint(file_management)
app.register_blueprint(stardist_controller)
app.register_blueprint(threshold)
app.register_blueprint(edge)
app.register_blueprint(hog)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
