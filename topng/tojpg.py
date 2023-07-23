import glob
import os
import cv2 as cv


if __name__ == '__main__':
    root_dir = '/home/garciamorenc/tfm/Real-ESRGAN/datasets/dataset_lq_train_600_x2/'
    result_dir = '/home/garciamorenc/tfm/Real-ESRGAN/datasets/dataset_lq_train_600_x2/'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.__contains__('roto'):
                result = cv.imread(os.path.join(subdir, file))
                new_file_name = file.replace('png', 'jpg')
                os.path.join(result_dir, new_file_name)
                new_file_name = os.path.join(result_dir, new_file_name)
                print(new_file_name)
                cv.imwrite(new_file_name, result)
