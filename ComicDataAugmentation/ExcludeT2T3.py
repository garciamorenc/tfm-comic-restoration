import cv2 as cv
import os
from PIL import Image
import torchvision.transforms.functional as TF
from DataAugmentation import DataAugmentation
from DataAugmentation2 import DataAugmentation2
import random

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    src_hr_train = '/home/garciamorenc/tfm/Real-ESRGAN/datasets/dataset_hr_train/'
    src_lq_train = '/home/garciamorenc/tfm/Real-ESRGAN/datasets/dataset_lq_train/'
    dest_hr_train = '/home/garciamorenc/tfm/Real-ESRGAN/datasets/dataset_hr_train_t2t3/'
    dest_lq_train = '/home/garciamorenc/tfm/Real-ESRGAN/datasets/dataset_lq_train_t2t3/'
    src_hr_test = '/home/garciamorenc/tfm/Real-ESRGAN/datasets/dataset_hr_test/'
    src_lq_test = '/home/garciamorenc/tfm/Real-ESRGAN/datasets/dataset_lq_test/'
    dest_hr_test = '/home/garciamorenc/tfm/Real-ESRGAN/datasets/dataset_hr_test_t2t3/'
    dest_lq_test = '/home/garciamorenc/tfm/Real-ESRGAN/datasets/dataset_lq_test_t2t3/'
    src_hr_validation = '/home/garciamorenc/tfm/Real-ESRGAN/datasets/dataset_hr_validation/'
    src_lq_validation = '/home/garciamorenc/tfm/Real-ESRGAN/datasets/dataset_lq_validation/'
    dest_hr_validation = '/home/garciamorenc/tfm/Real-ESRGAN/datasets/dataset_hr_validation_t2t3/'
    dest_lq_validation = '/home/garciamorenc/tfm/Real-ESRGAN/datasets/dataset_lq_validation_t2t3/'
    if not os.path.exists(dest_hr_train):
        os.makedirs(dest_hr_train)
    if not os.path.exists(dest_lq_train):
        os.makedirs(dest_lq_train)
    if not os.path.exists(dest_hr_validation):
        os.makedirs(dest_hr_validation)
    if not os.path.exists(dest_lq_validation):
        os.makedirs(dest_lq_validation)
    if not os.path.exists(dest_hr_test):
        os.makedirs(dest_hr_test)
    if not os.path.exists(dest_lq_test):
        os.makedirs(dest_lq_test)

    for filename in os.listdir(src_hr_train):
        if filename.__contains__('T2') or filename.__contains__('T3'):
            current = os.path.join(src_hr_train, filename)
            new = os.path.join(dest_hr_train, filename)
            os.rename(current, new)
            print(new)

    for filename in os.listdir(src_lq_train):
        if filename.__contains__('T2') or filename.__contains__('T3'):
            current = os.path.join(src_lq_train, filename)
            new = os.path.join(dest_lq_train, filename)
            os.rename(current, new)
            print(new)

    for filename in os.listdir(src_hr_validation):
        if filename.__contains__('T2') or filename.__contains__('T3'):
            current = os.path.join(src_hr_validation, filename)
            new = os.path.join(dest_hr_validation, filename)
            os.rename(current, new)
            print(new)

    for filename in os.listdir(src_lq_validation):
        if filename.__contains__('T2') or filename.__contains__('T3'):
            current = os.path.join(src_lq_validation, filename)
            new = os.path.join(dest_lq_validation, filename)
            os.rename(current, new)
            print(new)

    for filename in os.listdir(src_hr_test):
        if filename.__contains__('T2') or filename.__contains__('T3'):
            current = os.path.join(src_hr_test, filename)
            new = os.path.join(dest_hr_test, filename)
            os.rename(current, new)
            print(new)

    for filename in os.listdir(src_lq_test):
        if filename.__contains__('T2') or filename.__contains__('T3'):
            current = os.path.join(src_lq_test, filename)
            new = os.path.join(dest_lq_test, filename)
            os.rename(current, new)
            print(new)