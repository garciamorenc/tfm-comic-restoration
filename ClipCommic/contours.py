# This is a sample Python script.
import copy
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from argparse import ArgumentParser
import cv2
import numpy as np
import random as rng
import os
import glob
rng.seed(12345)


def equalize(path):
    """
    Detect text in a image
    :param path:  path to image
    """
    input = cv2.imread(path)
    img = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
    result = copy.deepcopy(input)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # cl1 = clahe.apply(img)
    # ath = cv2.adaptiveThreshold(cl1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 30)
    # th = cv2.bitwise_not(img)
    ret, th = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY_INV)    # edges = cv2.Canny(th, 127, 255)  # dan igual los umbrales porque el paso anterior lo deja en binario
    contours, hierarchy = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    drawing = np.zeros((th.shape[0], th.shape[1], 3), dtype=np.uint8)

    for i in range(len(contours)):
            epsilon = 0.001 * cv2.arcLength(contours[i], True)
            approx = cv2.approxPolyDP(contours[i], epsilon, True)
            # color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
            cv2.drawContours(drawing, [approx], -1, (0, 255, 0))
            # cv2.drawContours(drawing, approx, i, color, 2, cv2.LINE_8, hierarchy, 0)
            x, y, w, h = cv2.boundingRect(contours[i])
            cv2.rectangle(drawing, (x, y), (x + w, y + h), (0, 255, 255), 1)
            color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
            cv2.rectangle(result, (x, y), (x + w, y + h), color, cv2.FILLED)
    # Show in a window
    # stack = np.hstack((img, th))
    # stack2 = np.hstack((drawing, result))
    # cv2.namedWindow('finalImg', cv2.WINDOW_NORMAL)
    # cv2.imshow("finalImg", stack)
    # cv2.namedWindow('Proceso color', cv2.WINDOW_NORMAL)
    # cv2.imshow("Proceso color", stack2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    new_filename = os.path.basename(filename)
    cv2.imwrite(os.path.join('./results/', new_filename), result)

    # if len(contours) != 0:
    #     # draw in blue the contours that were founded
    #     cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    #     cv2.imshow("contours", img)
    #     cv2.waitKey(0)
    #
    #     # find the biggest countour (c) by the area
    #     # c = max(contours, key=cv2.contourArea)
    #     # x, y, w, h = cv2.boundingRect(c)
    #
    #     # draw the biggest contour (c) in green
    #     # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    folder = './inputs/'
    for filename in glob.glob(folder + '*.jpg'):
        equalize(filename)

