import cv2 as cv
import os
from PIL import Image
import torchvision.transforms.functional as TF
from DataAugmentation import DataAugmentation
from DataAugmentation2 import DataAugmentation2
import random

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    src = '/home/garciamorenc/tfm/dataset_comics/fdataset_hr_multiscale/'

    for filename in os.listdir(src):
        if filename.__contains__('T2') or filename.__contains__('T3'):
            current = os.path.join(src, filename)
            os.remove(current)
            print(current)