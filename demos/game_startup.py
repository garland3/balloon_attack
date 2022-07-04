from asyncio.windows_utils import pipe
from fileinput import filename
import sys
from pathlib import Path
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[1].as_posix())  # add kapao/ to path

import numpy as np
import torch
import argparse
import yaml
from utils.torch_utils import select_device
from utils.general import check_img_size, scale_coords
from utils.datasets import LoadImages, LoadWebcam
from utils.make_file_names import Make_file_names
from models.experimental import attempt_load
from val import run_nms, post_process_batch
import cv2
import os.path as osp
import json

def startup():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--img-path', default='res/crowdpose_100024.jpg', help='path to image')
    parser.add_argument('-o', '--out-path', default='output', help='output folder')


    # plotting options
    parser.add_argument('--bbox', action='store_true')
    parser.add_argument('--kp-bbox', action='store_true')
    parser.add_argument('--pose', action='store_true')
    parser.add_argument('--face', action='store_true')
    parser.add_argument('--color-pose', type=int, nargs='+', default=[255, 0, 255], help='pose object color')
    parser.add_argument('--color-kp', type=int, nargs='+', default=[0, 255, 255], help='keypoint object color')
    parser.add_argument('--line-thick', type=int, default=2, help='line thickness')
    parser.add_argument('--kp-size', type=int, default=1, help='keypoint circle size')
    parser.add_argument('--kp-thick', type=int, default=2, help='keypoint circle thickness')

    # model options
    parser.add_argument('--data', type=str, default='data/coco-kp.yaml')
    parser.add_argument('--imgsz', type=int, default=1280)
    parser.add_argument('--weights', default='kapao_l_coco.pt')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or cpu')
    parser.add_argument('--conf-thres', type=float, default=0.7, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--no-kp-dets', action='store_true', help='do not use keypoint objects')
    parser.add_argument('--conf-thres-kp', type=float, default=0.5)
    parser.add_argument('--conf-thres-kp-person', type=float, default=0.2)
    parser.add_argument('--iou-thres-kp', type=float, default=0.45)
    parser.add_argument('--overwrite-tol', type=int, default=25)
    parser.add_argument('--scales', type=float, nargs='+', default=[1])
    parser.add_argument('--flips', type=int, nargs='+', default=[-1])

    args = parser.parse_args()

    with open(args.data) as f:
        data = yaml.safe_load(f)  # load data dict

    # add inference settings to data dict
    data['imgsz'] = args.imgsz
    data['conf_thres'] = args.conf_thres
    data['iou_thres'] = args.iou_thres
    data['use_kp_dets'] = not args.no_kp_dets
    data['conf_thres_kp'] = args.conf_thres_kp
    data['iou_thres_kp'] = args.iou_thres_kp
    data['conf_thres_kp_person'] = args.conf_thres_kp_person
    data['overwrite_tol'] = args.overwrite_tol
    data['scales'] = args.scales
    data['flips'] = [None if f == -1 else f for f in args.flips]
    data['count_fused'] = False

    device = select_device(args.device, batch_size=1)
    print('Using device: {}'.format(device))

    model = attempt_load(args.weights, map_location=device)
    stride = int(model.stride.max())  # model stride
    print(f"stride = {stride}")
    imgsz = check_img_size(args.imgsz, s=stride)  # check image size

    return  args, data, model