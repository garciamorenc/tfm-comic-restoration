import cv2 as cv
import os
from PIL import Image
import torchvision.transforms.functional as TF
from DataAugmentation import DataAugmentation
from DataAugmentation2 import DataAugmentation2
import random

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    src = '/home/garciamorenc/tfm/dataset_comics/fdataset_hr_multiscale_sub/'
    dest = '/home/garciamorenc/tfm/dataset_comics/fdataset_hr_multiscale_sub_discard/'
    dest_600 = '/home/garciamorenc/tfm/dataset_comics/fdataset_hr_multiscale_sub_discard_600/'
    if not os.path.exists(dest):
        os.makedirs(dest)
    if not os.path.exists(dest_600):
        os.makedirs(dest_600)

    counter = 0
    for filename in os.listdir(src):
        img_path = os.path.join(src, filename)
        img = cv.imread(img_path)
        h, w = img.shape[:2]
        # if (h < 300 and w < 300) or (h < 250) or (w < 250):
        if (h < 600 or w < 600):
            counter = counter+1
            new_img_path = os.path.join(dest_600, filename)
            os.rename(img_path, new_img_path)

    print(counter)
