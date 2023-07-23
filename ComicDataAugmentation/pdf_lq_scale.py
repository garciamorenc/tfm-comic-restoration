import cv2 as cv
import os
from PIL import Image
import torchvision.transforms.functional as TF
from DataAugmentation import DataAugmentation
from DataAugmentation2 import DataAugmentation2
import random

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    src_lq_dir = '/home/garciamorenc/tfm/pdftoimage/roto/lq_unscale/'
    lq_dir = '/home/garciamorenc/tfm/pdftoimage/roto/lq/'
    if not os.path.exists(lq_dir):
        os.makedirs(lq_dir)

    w = None
    h = None
    for filename in os.listdir(src_lq_dir):
        if filename.__contains__('T1'):
            w = 1178
            h = 1690
        if filename.__contains__('T0'):
            w = 1756
            h = 2520
        img = cv.imread(os.path.join(src_lq_dir, filename))
        img = cv.resize(img, (w, h), interpolation=cv.INTER_AREA)
        new_filename = os.path.join(lq_dir, filename)
        print('lq: ', new_filename)
        cv.imwrite(new_filename, img)
