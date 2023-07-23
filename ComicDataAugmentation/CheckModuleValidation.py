

import cv2 as cv
import os
from PIL import Image
import torchvision.transforms.functional as TF
from DataAugmentation import DataAugmentation
from DataAugmentation2 import DataAugmentation2
import random
from basicsr import USMSharp
import numpy as np
import torchvision.transforms as transforms


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    src_hr_train = '/home/garciamorenc/tfm/Real-ESRGAN/datasets/fdataset_hr_train_600_lalala/'
    src_lq_train = '/home/garciamorenc/tfm/Real-ESRGAN/datasets/fdataset_lq_train_600_x2_lalala/'

    files = os.listdir(src_hr_train)
    sampled_list = random.sample(files, 999)

    for filename in sampled_list:
        current_hr = os.path.join(src_hr_train, filename)
        img_hr = cv.imread(current_hr)
        h_hr, w_hr = img_hr.shape[:2]

        if not h_hr != 600 and w_hr != 600:
            print('hr: ', h_hr, w_hr, current_hr)

        filename = filename.replace('.png', '.jpg')
        current_lq = os.path.join(src_lq_train, filename)
        img_lq = cv.imread(current_lq)
        h_lq, w_lq = img_lq.shape[:2]

        if not h_lq != 300 and w_lq != 300:
            print('lq: ', h_hr, w_hr, h_lq, w_lq, current_lq)




