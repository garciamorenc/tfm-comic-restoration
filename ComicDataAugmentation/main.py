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
    folder = '/home/garciamorenc/tfm/dataset_comics/fdataset_hr_multiscale_sub/'
    hr_dir = '/home/garciamorenc/tfm/Real-ESRGAN/datasets/fdataset_hr_train_600_lalala/'
    lq_dir = '/home/garciamorenc/tfm/Real-ESRGAN/datasets/fdataset_lq_train_600_x2_lalala/'
    # folder = '/home/garciamorenc/tfm/Real-ESRGAN/datasets/my_noise_hr_test/'
    # hr_dir = '/home/garciamorenc/tfm/Real-ESRGAN/datasets/dataset_hr_test_600/'
    # lq_dir = '/home/garciamorenc/tfm/Real-ESRGAN/datasets/dataset_lq_test_600_x2/'
    # folder = '/home/garciamorenc/tfm/Real-ESRGAN/datasets/my_noise_hr_validation/'
    # hr_dir = '/home/garciamorenc/tfm/Real-ESRGAN/datasets/dataset_hr_validation_600/'
    # lq_dir = '/home/garciamorenc/tfm/Real-ESRGAN/datasets/dataset_lq_validation_600_x2/'
    if not os.path.exists(hr_dir):
        os.makedirs(hr_dir)
    if not os.path.exists(lq_dir):
        os.makedirs(lq_dir)

    for filename in os.listdir(folder):
        if filename.__contains__('T2') or filename.__contains__('T3'):
            continue

        img = cv.imread(os.path.join(folder, filename))
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
            img = img[0:0 + crop_pad_size, 0:0 + crop_pad_size, ...]
        # crop
        if h >= crop_pad_size and w >= crop_pad_size:
            # randomly choose top and left coordinates
            top = random.randint(0, h - crop_pad_size)
            left = random.randint(0, w - crop_pad_size)
            img = img[top:top + crop_pad_size, left:left + crop_pad_size, ...]

        # cv2 to tensor
        transform = transforms.ToTensor()
        img = transform(img).unsqueeze_(0)
        # sharpen
        usm_sharpener = USMSharp()
        img = usm_sharpener(img)
        # tensor to cv2
        img = img.squeeze_(0)
        numpy_image = img.detach().numpy()
        img = np.transpose(numpy_image, (1, 2, 0))
        img = img*255

        new_filename = os.path.join(hr_dir, filename)
        new_filename = new_filename.replace('.jpg', '.png')
        print('hr: ', new_filename)
        cv.imwrite(new_filename, img)

    data_augmentation = DataAugmentation2()
    for filename in os.listdir(hr_dir):
        img = cv.imread(os.path.join(hr_dir, filename))
        result_img = data_augmentation.feed_data(img)
        new_filename = os.path.join(lq_dir, filename)
        new_filename = new_filename.replace('.png', '.jpg')
        print('lq', new_filename)
        cv.imwrite(new_filename, result_img)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
