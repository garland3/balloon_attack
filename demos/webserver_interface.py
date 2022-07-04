import requests
import numpy as np
import torch
import io


def get_server_response(img_torch):
    buff = io.BytesIO()
    torch.save(img_torch, buff)
    buff.seek(0)

    # data = img_np.to(torch._cast_Byte)
    res = requests.post(
        url="http://127.0.0.1:8000//something",
        data=buff.read(),
        headers={"Content-Type": "application/octet-stream"},
    )

    print(res.content)
    return res


if __name__ == "__main__":
    x = torch.rand(4, 4)
    get_server_response(x)
