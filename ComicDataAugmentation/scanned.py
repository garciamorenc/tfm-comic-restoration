import cv2 as cv
import os
from PIL import Image
import torchvision.transforms.functional as TF
from DataAugmentation import DataAugmentation

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    source_hr = '/home/garciamorenc/tfm/pdftoimage/scanned_hr_multiscale/'
    hr_dir = '/home/garciamorenc/tfm/pdftoimage/scanned_hr_multiscale_x4/'
    source_lq = '/home/garciamorenc/tfm/pdftoimage/scanned_lq/'
    source_lq_multiscale = '/home/garciamorenc/tfm/pdftoimage/scanned_lq_multiscale/'
    lq_dir = '/home/garciamorenc/tfm/pdftoimage/scanned_lq_multiscale_x4/'
    if not os.path.exists(hr_dir):
        os.makedirs(hr_dir)
    if not os.path.exists(lq_dir):
        os.makedirs(lq_dir)
    if not os.path.exists(source_lq_multiscale):
        os.makedirs(source_lq_multiscale)

    for filename in os.listdir(source_hr):
        img = cv.imread(os.path.join(source_hr, filename))
        h, w, _ = img.shape
        h = (h // 4) * 4
        w = (w // 4) * 4
        img = cv.resize(img, (w, h), interpolation=cv.INTER_AREA)
        new_filename = os.path.join(hr_dir, filename)
        print('hr x4: ', new_filename)
        cv.imwrite(new_filename, img)

    for filename in os.listdir(source_lq):
        img_lq = cv.imread(os.path.join(source_lq, filename))
        for i in range(0, 4):
            filename_multi = filename.replace('.png', 'T' + i.__str__() + '.png')
            new_filename = os.path.join(source_lq_multiscale, filename_multi)
            print('lq multi: ', new_filename)
            cv.imwrite(new_filename, img_lq)

    for filename in os.listdir(hr_dir):
        img_hr = cv.imread(os.path.join(hr_dir, filename))
        h, w, _ = img_hr.shape
        h = (h // 4)
        w = (w // 4)
        img_lq = cv.imread(os.path.join(source_lq_multiscale, filename))
        img_lq = cv.resize(img_lq, (w, h), interpolation=cv.INTER_AREA)
        new_filename = os.path.join(lq_dir, filename)
        print('lq x4: ', new_filename)
        cv.imwrite(new_filename, img_lq)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
