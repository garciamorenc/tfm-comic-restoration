# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from argparse import ArgumentParser
import cv2
import numpy as np
import random as rng


rng.seed(12345)


def image(path):
    """
    Detect text in a image
    :param path:  path to image
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # equ = cv2.equalizeHist(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(img)
    # cv2.imshow("Text detection result", img)
    # res = np.hstack((img, equ))  # stacking images side-by-side

    # thresh = 127
    # im_bw = cv2.threshold(cl1, thresh, 255, cv2.THRESH_BINARY)[1]
    edges = cv2.Canny(cl1, 200, 250)
    cv2.imshow("im_bw", edges)
    # cv2.imwrite("im_bw.jpg", edges)
    cv2.waitKey(0)


def thresh_callback(val):
    threshold = val
    # Detect edges using Canny
    canny_output = cv2.Canny(src_gray, threshold, threshold * 2)
    # Find contours
    contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Draw contours
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    for i in range(len(contours)):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv2.drawContours(drawing, contours, i, color, 2, cv2.LINE_8, hierarchy, 0)
    # Show in a window
    cv2.imshow('Contours', drawing)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # parser = ArgumentParser(description='Code for Histogram Equalization tutorial.')
    # parser.add_argument('--input', help='Path to input image.', default='lena.jpg')
    # args = parser.parse_args()
    # image('lena.jpg')
    # image('comic.jpg')

    src = cv2.imread('comic.jpg')
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    src_gray = cv2.blur(src_gray, (3, 3))
    # Create Window
    source_window = 'Source'
    cv2.namedWindow(source_window)
    cv2.imshow(source_window, src)
    max_thresh = 255
    thresh = 100  # initial threshold
    cv2.createTrackbar('Canny Thresh:', source_window, thresh, max_thresh, thresh_callback)
    thresh_callback(thresh)
    cv2.waitKey()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/