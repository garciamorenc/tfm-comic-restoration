# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from argparse import ArgumentParser
import cv2
import numpy as np


def equalize(path):
    """
    Detect text in a image
    :param path:  path to image
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # equ = cv2.equalizeHist(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(img)
    thresh = 127
    # im_bw1 = cv2.threshold(cl1, 200, 255, cv2.THRESH_BINARY)[1]
    # im_bw2 = cv2.threshold(cl1, 200, 255, cv2.THRESH_OTSU)[1]
    th1 = cv2.adaptiveThreshold(cl1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 20)
    th2 = cv2.adaptiveThreshold(cl1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2)
    th3 = cv2.adaptiveThreshold(cl1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 20)
    th4 = cv2.adaptiveThreshold(cl1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    th5 = cv2.adaptiveThreshold(cl1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 60)
    stack = np.hstack((cl1, th1, th2, th3, th4, th5))
    cv2.imshow("Equalize histogram", stack)
    cv2.waitKey(0)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    equalize('comic.jpg')

