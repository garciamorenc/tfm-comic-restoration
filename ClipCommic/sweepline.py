import cv2 as cv
import numpy as np
import rtree
from Panel import Panel


def init_structure():
    x_roi, y_roi, w_roi, h_roi = 10, 10, 320, 320
    rectangle_roi = (x_roi, y_roi, w_roi, h_roi)
    rectangle1 = (20, 100, 70, 150)
    rectangle2 = (80, 22, 280, 162)
    rectangle3 = (200, 300, 270, 310)
    rectangle4 = (300, 40, 310, 240)
    rectangle5 = (90, 20, 150, 70)
    rectangles = [rectangle1, rectangle2, rectangle3, rectangle4, rectangle5]
    return rectangles


def show_rectangles(rectangles):
    img = np.zeros((350, 350, 3), dtype=np.uint8)

    for i in range(len(rectangles)):
        x, y, w, h = rectangles[i]
        color = (0, 0, 255)
        if i == 0:
            color = (255, 255, 255)

        cv.rectangle(img, (x, y), (w, h), color, 1)
    cv.imshow('black image', img)
    cv.waitKey()


def group_by_intersections(rectangles):
    idx = rtree.index.Index(interleaved=True)
    for i in range(len(rectangles)):
        idx.insert(i, rectangles[i])

    groups = []
    # print(rectangles)
    for rectangle in rectangles:
        hits = list(idx.intersection(rectangle))

        flag = False
        length = len(groups)
        for k in range(length):
            if not set(hits).isdisjoint(groups[k]):
                # print("union")
                # print(groups)
                # print(hits)
                # print(list(set(groups[k] + hits)))
                groups[k] = list(set(groups[k] + hits))
                flag = False
                break
            else:
                flag = True
            # print(groups)

        if flag or length == 0:
            groups.append(hits)

    return groups


def group_by(rectangles):
    idx = rtree.index.Index(interleaved=True)
    for i in range(len(rectangles)):
        idx.insert(i, rectangles[i])

    groups = []
    for rectangle in rectangles:
        hits = list(idx.intersection(rectangle))

        flag = False
        length = len(groups)
        for k in range(length):
            if not set(hits).isdisjoint(groups[k].index_intersections):
                a = np.array(rectangles)
                b = hits
                groups[k].__add__(hits, list(a[b]))
                flag = False
                break
            else:
                flag = True

        if flag or length == 0:
            a = np.array(rectangles)
            b = hits
            # test = list(a[b])
            panel = Panel(hits, list(a[b]))
            # print('test' + str(np.array(test).max(initial=0, axis=0)))
            # print(test)
            # panel = Panel(hits, list(a[b]))
            groups.append(panel)

    return groups


if __name__ == '__main__':
    rectangles_list = init_structure()
    show_rectangles(rectangles_list)
    results = group_by(rectangles_list)

    img_result = np.zeros((350, 350, 3), dtype=np.uint8)
    for result in results:
        print("final index: " + str(result.index_intersections) + "final coords: " +
              str(result.x_min) + " " + str(result.y_min) + " " + str(result.x_max) + " " + str(result.y_max))
        cv.rectangle(img_result, (result.x_min, result.y_min), (result.x_max, result.y_max), (255, 0, 0), 1)
        cv.imshow('result image', img_result)

    cv.waitKey()
    cv.destroyAllWindows()