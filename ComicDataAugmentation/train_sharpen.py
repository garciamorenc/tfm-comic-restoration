import cv2 as cv
import os
from PIL import Image
import torchvision.transforms.functional as TF
from basicsr import USMSharp

from DataAugmentation import DataAugmentation
from DataAugmentation2 import DataAugmentation2
import random
import numpy as np
import torchvision.transforms as transforms

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    filename = '/home/garciamorenc/tfm/Real-ESRGAN/datasets/sharpen_hr_their/img1039T2.png'
    new_filename_hr = '/home/garciamorenc/tfm/Real-ESRGAN/datasets/sharpen_hr_their/mine_gt_img.png'
    new_filename_lq = '/home/garciamorenc/tfm/Real-ESRGAN/datasets/sharpen_hr_their/mine_lq_img.jpg'

    img = cv.imread(filename)
    h, w, _ = img.shape
    h = (h // 2) * 2
    w = (w // 2) * 2
    img = cv.resize(img, (w, h), interpolation=cv.INTER_AREA)
    # crop or pad to 400
    crop_pad_size = 600
    # pad
    if h < crop_pad_size or w < crop_pad_size:
        pad_h = max(0, crop_pad_size - h)
        pad_w = max(0, crop_pad_size - w)
        img = cv.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv.BORDER_REFLECT_101)
    # crop
    if h >= crop_pad_size and w >= crop_pad_size:
        # randomly choose top and left coordinates
        top = random.randint(0, h - crop_pad_size)
        left = random.randint(0, w - crop_pad_size)
        img = img[top:top + crop_pad_size, left:left + crop_pad_size, ...]

    transform = transforms.ToTensor()
    img = transform(img).unsqueeze_(0)
    usm_sharpener = USMSharp()
    img = usm_sharpener(img)
    img = img.squeeze_(0)
    numpy_image = img.detach().numpy()
    img = np.transpose(numpy_image, (1, 2, 0))
    img = img*255

    cv.imwrite(new_filename_hr, img)

    data_augmentation = DataAugmentation2()
    img = cv.imread(new_filename_hr)
    result_img = data_augmentation.feed_data(img)
    new_filename = new_filename_lq
    cv.imwrite(new_filename, result_img)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
