# import Flask
from flask import Flask, request
# import request
# 
# s
import torch
import io
import numpy as np
from PIL import Image
app = Flask(__name__)


@app.route("/something", methods=["POST", "GET"])
def process_img():
    bytes_image = request.files.get('bytes_image')
    bytes_image = bytes_image.read()
    dtype = np.float32
    array_image = np.frombuffer(bytes_image, dtype=dtype)
    shape = (int(request.form['image_shape_width']), int(request.form['image_shape_height']))         
    array_image = np.reshape(array_image, shape)
    print(array_image)
                                    
    # image = Image.fromarray(array_image)
    # print(image)
    return "ok"


@app.route("/")
def index():
    return "Hello World Test"


if __name__ == "__main__":
    app.run(debug=True, threaded=True, port=8000)
