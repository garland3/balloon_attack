# Import random for random numbers

from pathlib import Path
import sys

# from demos.webserver_interface import get_server_response

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[1].as_posix())  # add kapao/ to path

import random

# from asyncio.windows_utils import pipe
from fileinput import filename

import numpy as np
import torch
import time
import json

import cv2
from utils.datasets import LoadImages, LoadWebcam

from demos.game_startup import startup
from val import run_nms, post_process_batch
from utils.torch_utils import select_device


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

args, data, model = startup()
args.pose = True
device = select_device(args.device, batch_size=1)


@app.route("/something", methods=["POST", "GET"])
def process_img():
    bytes_image = request.files.get('bytes_image')
    bytes_image = bytes_image.read()
    dtype = np.float32
   
    shape = (int(request.form['image_shape_channels']),
        int(request.form['image_shape_width']), int(request.form['image_shape_height'])) 
    array_image = np.frombuffer(bytes_image, dtype=dtype, count  = shape[0]*shape[1]*shape[2])
    print("shape", shape)        
    array_image = np.reshape(array_image, shape)
    # print(array_image)
                                    
    # image = Image.fromarray(array_image)
    # print(image)
    array_image = torch.tensor(array_image)
    if len(array_image.shape) == 3:
        array_image = array_image[None]  # expand for batch dim

    out = model(
            array_image,
            augment=True,
            kp_flip=data["kp_flip"],
            scales=data["scales"],
            flips=data["flips"],
        )[0]
    person_dets, kp_dets = run_nms(data, out)

    # print("person_dets looping type")
    # print([type(d) for d in person_dets ])

    print("person dets shape", person_dets[0].shape)
    print("kp_dets shape", kp_dets[0].shape)

    person_dets_listish = person_dets[0].to(torch.float32).tolist()
    kp_dets_listish = kp_dets[0].to(torch.float32).tolist()
    # print("person_dets", person_dets)
    # print("kp_dets", kp_dets)

    return json.dumps(
        {"person_dets":person_dets_listish, 
        "person_dets_shape":person_dets[0].shape, 
        "kp_dets":kp_dets_listish, 
        "kp_dets_shape":kp_dets[0].shape}, default = str
    )


@app.route("/")
def index():
    return "Hello World Test"


if __name__ == "__main__":
    app.run(debug=True, threaded=True, port=8000)
