from flask import Flask
from controller.file_management_controller import file_management
from controller.stardist_controller import stardist_controller
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

app = Flask(__name__)

app.register_blueprint(file_management)
app.register_blueprint(stardist_controller)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
