import cv2 as cv
import numpy as np
import rtree
from Panel import Panel


def init_structure():
    x_roi, y_roi, w_roi, h_roi = 0, 0, 1920, 1080
    rectangle_roi = (x_roi, y_roi, w_roi, h_roi)
    rectangle1 = (0, 0, 192, 108)
    rectangle2 = (0, 0, 147, 83)
    rectangle3 = (0, 0, 128, 72)
    rectangle4 = (0, 0, 96, 54)
    rectangles = [rectangle_roi,  rectangle1, rectangle2, rectangle3, rectangle4]
    return rectangles


def show_rectangles(rectangles):
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)

    for i in range(len(rectangles)):
        x, y, w, h = rectangles[i]
        color = (0, 0, 255)
        if i == 0:
            color = (255, 255, 255)

        cv.rectangle(img, (x, y), (w, h), color, 1)
    cv.imshow('black image', img)
    cv.waitKey()



if __name__ == '__main__':
    rectangles_list = init_structure()
    show_rectangles(rectangles_list)
    cv.waitKey()
    cv.destroyAllWindows()