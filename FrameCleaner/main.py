import glob
import os
import cv2 as cv

h_even = 130
w_even = 130

y_odd = 130
x_odd = 130


def comic_crop(image_path):
    last_char = int(image_path[-5])
    img = cv.imread(image_path)
    h, w, _ = img.shape

    if (last_char % 2) == 0:
        crop_img = img[0:0 + (h - h_even), 0:0 + (w - w_even)]
    else:
        crop_img = img[y_odd:y_odd + h, x_odd:x_odd + w]

    return crop_img
    # cv.imshow("cropped", crop_img)
    # cv.waitKey(0)


if __name__ == '__main__':
    root_dir = '/home/garciamorenc/tfm/dataset_comics/'
    result_dir = '/home/garciamorenc/tfm/dataset_comics/results/'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    for subdir, dirs, files in os.walk(root_dir):
        if 'result' in subdir:
            continue
        for file in files:
            if 'cuphead' not in subdir:
                result = comic_crop(os.path.join(subdir, file))
            else:
                result = cv.imread(os.path.join(subdir, file))

            comic_name = subdir.split('/')[-1]
            new_file_name = comic_name + '_' + file[-8:]
            new_file_name = os.path.join(result_dir, new_file_name)
            print(new_file_name)
            cv.imwrite(new_file_name, result)
