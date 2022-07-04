from asyncio.windows_utils import pipe
from fileinput import filename
import sys
from pathlib import Path
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[1].as_posix())  # add kapao/ to path

import numpy as np
import torch
import time

import cv2
from utils.datasets import LoadImages, LoadWebcam

step_time = 0.5

def main():
    imgsz = 256
    stride = 64
    dataset = LoadWebcam("0", imgsz, stride )

    last_step = time.perf_counter()

    for i in range(len(dataset)):
        (_, img, im0, _) = next(iter(dataset))
        print(f"step {i}")

        # Display the resulting frame
        cv2.imshow('frame', im0)

        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            dataset.ext()
            break

        current_step = time.perf_counter()
        elpased_time = current_step - last_step
        last_step = current_step
        # print(f"elpased time = {elpased_time}")
        if elpased_time<step_time:
            time.sleep(1 - elpased_time)
            # print(f"Slowing down. Elapsed time was {elpased_time}, but step is {step_time}")

    # Destroy all the windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()