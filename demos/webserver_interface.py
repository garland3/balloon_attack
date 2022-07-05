import requests
import numpy as np
import torch
import io
import http

ip =  "http://127.0.0.1:8000//something"

def get_server_response(img_torch):
    x_np = img_torch.numpy()
    info = dict()
    info["image_shape_width"] = x_np.shape[0]
    info["image_shape_height"] =  x_np.shape[1]

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

    # print(res.content)
    # return res


if __name__ == "__main__":
    x = torch.rand((4, 4))
    print(x, x.dtype)
    get_server_response(x)
