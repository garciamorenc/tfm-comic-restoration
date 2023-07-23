import cv2 as cv
import os
from PIL import Image
import torchvision.transforms.functional as TF
from DataAugmentation import DataAugmentation
from DataAugmentation2 import DataAugmentation2
import random
from basicsr import USMSharp
import torchvision.transforms as transforms
import numpy as np

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    x2_dir = '/home/garciamorenc/tfm/Real-ESRGAN/datasets/fdataset_lq_test_600_x2_x2/'
    x4_dir = '/home/garciamorenc/tfm/Real-ESRGAN/datasets/matlab_x4'
    if not os.path.exists(x4_dir):
        os.makedirs(x4_dir)

    for filename in os.listdir(x2_dir):
        img = cv.imread(os.path.join(x2_dir, filename))
        h, w, _ = img.shape
        scale_dim = (int(w/2), int(h/2))
        out = cv.resize(img, scale_dim, interpolation=cv.INTER_AREA)
        cv.imwrite(os.path.join(x4_dir, filename), out)
        print(filename)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
