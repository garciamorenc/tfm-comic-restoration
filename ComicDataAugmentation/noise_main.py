import cv2
import cv2 as cv
import os
from PIL import Image
import torchvision.transforms.functional as TF
from DataAugmentation import DataAugmentation
from DataAugmentation2 import DataAugmentation2

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    hr_dir = '/home/garciamorenc/tfm/ComicDataAugmentation/morta_hr/'
    lq_dir = '/home/garciamorenc/tfm/ComicDataAugmentation/morta_lq'
    if not os.path.exists(lq_dir):
        os.makedirs(lq_dir)

    data_augmentation = DataAugmentation2()
    for filename in os.listdir(hr_dir):
        img = cv.imread(os.path.join(hr_dir, filename))
        result_img = data_augmentation.feed_data(img)
        new_filename = os.path.join(lq_dir, filename)
        print('lq', new_filename)
        # cv.imwrite(new_filename, result_img)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
