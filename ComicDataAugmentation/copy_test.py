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
    src_hr_train = '/home/garciamorenc/tfm/Real-ESRGAN/datasets/fdataset_hr_train_600/'
    src_lq_train = '/home/garciamorenc/tfm/Real-ESRGAN/datasets/fdataset_lq_train_600_x2/'
    dest_hr_dir = '/home/garciamorenc/tfm/Real-ESRGAN/datasets/fdataset_hr_test_600/'
    dest_lq_dir = '/home/garciamorenc/tfm/Real-ESRGAN/datasets/fdataset_lq_test_600_x2/'
    if not os.path.exists(dest_hr_dir):
        os.makedirs(dest_hr_dir)
    if not os.path.exists(dest_lq_dir):
        os.makedirs(dest_lq_dir)

    files = os.listdir(src_hr_train)
    sampled_list = random.sample(files, 999)

    for filename in sampled_list:
        current = os.path.join(src_hr_train, filename)
        new = os.path.join(dest_hr_dir, filename)
        os.rename(current, new)
        print(new)

        current_lq = os.path.join(src_lq_train, filename)
        if not os.path.isfile(current_lq):
            current_lq = current_lq.replace('.png', '.jpg')
        new_lq = os.path.join(dest_lq_dir, filename)
        os.rename(current_lq, new_lq)
        print(new_lq)

    # for filename in os.listdir(src_test):
    #     current = os.path.join(src_hr_train, filename)
    #     if not os.path.isfile(current):
    #         current = current.replace('.jpg', '.png')
    #     new = os.path.join(dest_hr_dir, filename)
    #     os.rename(current, new)
    #     print(new)
    #
    #     current = os.path.join(src_lq_train, filename)
    #     if not os.path.isfile(current):
    #         current = current.replace('.jpg', '.png')
    #     new = os.path.join(dest_lq_dir, filename)
    #     os.rename(current, new)
    #     print(new)