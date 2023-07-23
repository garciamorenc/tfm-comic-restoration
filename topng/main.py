import glob
import os
import cv2 as cv


if __name__ == '__main__':
    root_dir = '/home/garciamorenc/tfm/pdftoimage/roto/'
    result_dir = '/home/garciamorenc/tfm/dataset_comics/roto_png/'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            result = cv.imread(os.path.join(subdir, file))
            new_file_name = file.replace('tif', 'png')
            os.path.join(result_dir, new_file_name)
            new_file_name = os.path.join(result_dir, new_file_name)
            print(new_file_name)
            cv.imwrite(new_file_name, result)
