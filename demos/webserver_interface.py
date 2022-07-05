import requests
import numpy as np
import torch
import io
import http
import json
ip =  "http://127.0.0.1:8000//something"

def get_server_response(img_np):
    print("img type", type(img_np))
    if type(img_np) == torch.Tensor:
        img_np = img_np.numpy()
    print("image shape is ", img_np.shape)
    x_np = img_np #.numpy()
    info = dict()
    info["image_shape_channels"] =  x_np.shape[0]
    info["image_shape_width"] = x_np.shape[1]
    info["image_shape_height"] =  x_np.shape[2]


    bytes_image = x_np.tobytes()
    stream = io.BytesIO(bytes_image)
    files = {"bytes_image": stream}

    info["array_image"] = None

    # response = http.post(ip + "path", data=info, files=files)
    # buff = io.BytesIO()
    # torch.save(img_torch, buff)
    # buff.seek(0)

    # # data = img_np.to(torch._cast_Byte)
    res = requests.post(
        url="http://127.0.0.1:8000//something",
        data= info, 
        files=files,
        # headers={"Content-Type": "application/octet-stream"},
    )


    package = json.loads( res.content)
    print("package", package)
    person_dets, kp_dets  = package['person_dets'], package['kp_dets']
    person_dets_shape, kp_dets_shape  = package['person_dets_shape'], package['kp_dets_shape']

    person_dets = [torch.Tensor(d).to(torch.float32) for d in person_dets]
    kp_dets = [torch.Tensor(d).to(torch.float32) for d in kp_dets]


    person_dets = [torch.cat(person_dets).view(person_dets_shape)]
    kp_dets = [torch.cat(kp_dets).view(kp_dets_shape)]


    return person_dets, kp_dets
    # print(res.content)
    # return res


if __name__ == "__main__":
    x =np.random.rand(3,5,5).astype(np.float32)
    print(x, x.dtype)
    get_server_response(x)
