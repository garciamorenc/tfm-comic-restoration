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
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(img)
    th = cv2.adaptiveThreshold(cl1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 60)
    edges1 = cv2.Canny(th, 127, 255)  # dan igual los umbrales porque el paso anterior lo deja en binario
    stack = np.hstack((cl1, th, edges1))
    cv2.imshow("Equalize histogram", stack)
    cv2.waitKey(0)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    equalize('comic.jpg')

