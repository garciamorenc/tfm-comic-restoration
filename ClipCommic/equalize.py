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
    stack = np.hstack((img, cl1))
    cv2.imshow("Equalize histogram", stack)
    cv2.waitKey(0)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    equalize('comic.jpg')


