from PIL import Image
import argparse
from cv2 import threshold
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
# from torch import int32

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", '-i', dest="image", help="image to act on")
    p.add_argument("--task", '-t', dest="task", help="task", default="clean")

    args = p.parse_args()
    im = Image.open(args.image)
    im_np = np.array(im)
    sz = im_np.shape
    f_original = Path(args.image)


    if args.task == 'mirror':
        img2 = np.zeros((sz[0], sz[1]*2,3)).astype(int)
        img2[:,0:sz[1],:] = im_np
        img2[:,sz[1]:,:] =  np.flip(im_np,1)
        fname = "m_" + f_original.name
        f_full = f_original.parent / fname
       
    if args.task == 'clean':
        threshold = 220*3
        mask = im_np.sum(axis = 2) > threshold
        # plt.imshow(mask)
        # plt.show()
        # plt.
        im_np[mask, :] = np.array([255,255,255])
        img2 = im_np
        plt.imshow(img2)
        plt.show()

        fname = "clean_" + f_original.name
        f_full = f_original.parent / fname


    image_img2 = Image.fromarray(img2.astype('uint8'), 'RGB')
    image_img2.save(f_full)
    print(f_full)
main()