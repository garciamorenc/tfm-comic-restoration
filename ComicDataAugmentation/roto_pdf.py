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
    # hr_dir = '/home/garciamorenc/tfm/Real-ESRGAN/datasets/dataset_hr_train/'
    # lq_dir = '/home/garciamorenc/tfm/Real-ESRGAN/datasets/dataset_lq_train/'
    # hr_dir = '/home/garciamorenc/tfm/Real-ESRGAN/datasets/dataset_hr_test/'
    # lq_dir = '/home/garciamorenc/tfm/Real-ESRGAN/datasets/dataset_lq_test/'
    src_hr_dir = '/home/garciamorenc/tfm/Real-ESRGAN/datasets/dataset_hr_train_600/'
    src_lq_dir = '/home/garciamorenc/tfm/Real-ESRGAN/datasets/dataset_lq_train_600_x2/'
    dest_hr_dir = '/home/garciamorenc/tfm/Real-ESRGAN/datasets/dataset_hr_train_600/'
    dest_lq_dir = '/home/garciamorenc/tfm/Real-ESRGAN/datasets/dataset_lq_train_600_x2/'
    if not os.path.exists(dest_hr_dir):
        os.makedirs(dest_hr_dir)
    if not os.path.exists(dest_lq_dir):
        os.makedirs(dest_lq_dir)

    for filename in os.listdir(src_hr_dir):
        is_roto = filename.__contains__("roto")
        is_t2t3 = filename.__contains__('T2') or filename.__contains__('T3')
        # is_cafe = filename.__contains__("cafe")
        # is_roto = filename.__contains__("roto")
        if is_roto and not is_t2t3:
            img = cv.imread(os.path.join(src_hr_dir, filename))
            img_lq = cv.imread(os.path.join(src_lq_dir, filename))

            # resize lq x2
            h, w, _ = img.shape
            h_lq = (h // 2)
            w_lq = (w // 2)
            img_lq = cv.resize(img_lq, (w_lq, h_lq), interpolation=cv.INTER_AREA)

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
            img = img * 255

            new_filename = os.path.join(dest_hr_dir, filename)
            print('hr: ', new_filename)
            cv.imwrite(new_filename, img)

            new_filename = os.path.join(dest_lq_dir, filename)
            os.remove(new_filename)
            new_filename = new_filename.replace('.png', '.jpg')
            print('lq: ', new_filename)
            cv.imwrite(new_filename, img_lq)
