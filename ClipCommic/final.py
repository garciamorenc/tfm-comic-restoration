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
import rtree
from scipy.spatial import distance
from Panel import Panel

rng.seed(12345)


def find_contours(path):
    """
    Detect contours in an image
    :param path:  path to image
    """
    input = cv2.imread(path)
    img = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
    result = copy.deepcopy(input)
    # Creating kernel
    wones = np.ones((1, 3), np.uint8)
    hones = np.ones((1, 3), np.uint8)
    zeros = np.zeros((5, 5), np.uint8)
    # Using cv2.erode() method
    ret, th = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY_INV)
    th = cv2.erode(th, wones, iterations=1)
    th = cv2.erode(th, hones, iterations=1)
    th = cv2.dilate(th, wones, iterations=1)
    th = cv2.dilate(th, hones, iterations=1)
    # th = cv2.Canny(img, 127, 255)  # dan igual los umbrales porque el paso anterior lo deja en binario
    contours, hierarchy = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    drawing = np.zeros((th.shape[0], th.shape[1], 3), dtype=np.uint8)
    return contours, hierarchy, img, result, th


def find_panels(contours):
    my_list = []
    for i in range(len(contours)):
        epsilon = 0.001 * cv2.arcLength(contours[i], True)
        approx = cv2.approxPolyDP(contours[i], epsilon, True)
        # color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
        # cv2.drawContours(drawing, [approx], -1, (0, 255, 0))
        # cv2.drawContours(drawing, approx, i, color, 2, cv2.LINE_8, hierarchy, 0)
        x, y, w, h = cv2.boundingRect(approx)
        my_list.append((x, y, (x + w), (y + h)))

    return my_list


def group_by_panels_candidates(rectangles):
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
    return groups, idx


def calculate_size_panels(panels, image_size):
    for panel in panels:
        panel.__size_classifier__(image_size)


def area(a, b):
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    return dx * dy


def rect_distance(x1, y1, x1b, y1b, x2, y2, x2b, y2b):
    left = x2b < x1
    right = x1b < x2
    bottom = y2b < y1
    top = y1b < y2
    if top and left:
        a = (x1, y1b)
        b = (x2b, y2)
        return np.linalg.norm(a-b)
    elif left and bottom:
        a = (x1, y1)
        b = (x2b, y2b)
        return np.linalg.norm(a - b)
    elif bottom and right:
        a = (x1b, y1)
        b = (x2, y2b)
        return np.linalg.norm(a-b)
    elif right and top:
        a = (x1b, y1b)
        b = (x2, y2)
        return np.linalg.norm(a - b)
    elif left:
        return x1 - x2b
    elif right:
        return x2 - x1b
    elif bottom:
        return y1 - y2b
    elif top:
        return y2 - y1b
    else:             # rectangles intersect
        return 0.


def group_by_medium_panels_candidates(rectangles, image_size):
    image_size_w, image_size_h = image_size
    minimun_distance = image_size_h * 1/15
    if image_size_w > image_size_h:
        minimun_distance = image_size_w * 1/15
    #TODO calcular cual es más grande y aplicar un % de reducción
    idx_medium = rtree.index.Index(interleaved=True)
    for i in range(len(rectangles)):
        idx_medium.insert(i, rectangles[i])

    groups = []
    for rectangle in rectangles:
        near_hits = list(idx_medium.nearest(rectangle, 10))

        for i in range(len(near_hits)-1):
            center_distance = calculate_center(rectangle, rectangles[near_hits[i+1]])
            flag = False
            if center_distance < minimun_distance:  # 70
                length = len(groups)
                test = [near_hits[0], near_hits[i+1]]
                for k in range(length):
                    if not set(test).isdisjoint(groups[k].index_intersections):
                        a = np.array(rectangles)
                        b = test
                        groups[k].__add__(test, list(a[b]))
                        flag = False
                        break
                    else:
                        flag = True

                if flag or length == 0:
                    a = np.array(rectangles)
                    b = test
                    # test = list(a[b])
                    panel = Panel(test, list(a[b]))
                    # print('test' + str(np.array(test).max(initial=0, axis=0)))
                    # print(test)
                    # panel = Panel(hits, list(a[b]))
                    groups.append(panel)

        # group_results = []
        # for i in range(len(groups)):
        #     if i == 0:
        #         group_results.append(groups[i])
        #     for k in range(len(group_results)):
        #         if not set(groups[i].index_intersections).isdisjoint(groups[k].index_intersections):
        #             a = np.array(rectangles)
        #             b = groups[i].index_intersections
        #             groups[k].__add__(b, list(a[b]))
    return groups


def calculate_center(r1, r2):
    r1_x_min, r1_y_min, r1_x_max, r1_y_max = r1
    r2_x_min, r2_y_min, r2_x_max, r2_y_max = r2
    x_center_r1 = (r1_x_min + r1_x_max) / 2
    y_center_r1 = (r1_y_min + r1_y_max) / 2
    x_center_r2 = (r2_x_min + r2_x_max) / 2
    y_center_r2 = (r2_y_min + r2_y_max) / 2
    w_r1 = r1_x_max - r1_x_min
    w_r2 = r2_x_max - r2_x_min
    b_r1 = r1_y_max - r1_y_min
    b_r2 = r2_y_max - r2_y_min
    center_distance = max(abs(x_center_r1 - x_center_r2) - (w_r1 + w_r2)/2, abs(y_center_r1 - y_center_r2) - (b_r1 + b_r2)/2)
    return center_distance


def run_clip(filename):
    result = []
    my_contours, my_hierarchy, my_img, my_img_copy, my_th = find_contours(filename)
    my_panels_init = find_panels(my_contours)

    # Eliminamos los bbox despreciables por tamaño 1/100 del total de la imagen
    image_size = my_img.shape[:2]
    for x, y, w, h in my_panels_init:  # refactorizar a utilizar clasificador de paneles calculate_size_panels
        if (w - x < image_size[0] * 1 / 100) and (h - y < image_size[1] * 1 / 100):
            my_panels_init.remove((x, y, w, h))

    my_panels, my_tree_idx = group_by_panels_candidates(my_panels_init)
    calculate_size_panels(my_panels, my_img_copy.shape[:2])  # TODO: buscar pequeñas
    for bbox in my_panels:
        if bbox.size_classification == 'big':
            result.append(bbox)
            # color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
            # cv2.rectangle(my_img_copy, (bbox.x_min, bbox.y_min), (bbox.x_max, bbox.y_max), color, 5) # Debug
        # elif bbox.size_classification == 'medium':
        #     color = (0, 0, 255)

    # my_tree_idx = rtree.index.Index(interleaved=True)
    # for i in range(len(my_panels)):
    #     my_tree_idx.insert(i, my_panels[i])
    # https://math.stackexchange.com/questions/2724537/finding-the-clear-spacing-distance-between-two-rectangles
    my_panels_medium = [(my_panels[i].x_min, my_panels[i].y_min, my_panels[i].x_max, my_panels[i].y_max) for i in
                        range(len(my_panels)) if my_panels[i].size_classification != 'big']
    my_panels_medium = group_by_medium_panels_candidates(my_panels_medium, image_size)
    my_panels_medium = [
        (my_panels_medium[i].x_min, my_panels_medium[i].y_min, my_panels_medium[i].x_max, my_panels_medium[i].y_max) for
        i in
        range(len(my_panels_medium))]
    my_panels_medium, my_tree_idx_medium = group_by_panels_candidates(my_panels_medium)
    for bbox in my_panels_medium:
        result.append(bbox)
        # color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
        # cv2.rectangle(my_img_copy, (bbox.x_min, bbox.y_min), (bbox.x_max, bbox.y_max), (255,255,0), 5) # Debug


    # for panel in my_panels:
    #     if panel.size_classification == 'medium':
    #         near_hit = list(my_tree_idx.nearest((panel.x_min, panel.y_min, panel.x_max, panel.y_max), 1))[0]
    #         bbox_near_hit_x_min, bbox_near_hit_y_min, bbox_near_hit_x_max, bbox_near_hit_y_max = my_panels_init[near_hit]
    #         xCenter1 = (panel.x_min + panel.x_max) / 2
    #         yCenter1 = (panel.y_min + panel.y_max) / 2
    #         xCenter2 = (bbox_near_hit_x_min + bbox_near_hit_x_max) / 2
    #         yCenter2 = (bbox_near_hit_y_min + bbox_near_hit_y_max) / 2
    #
    #         w1 = panel.x_max - panel.x_min
    #         w2 = bbox_near_hit_x_max - bbox_near_hit_x_min
    #         b1 = panel.y_min - panel.x_min
    #         b2 = bbox_near_hit_y_min - bbox_near_hit_x_min
    #
    #         distance = max(abs(xCenter1 - xCenter2) - (w1 + w2)/2, abs(yCenter1 - yCenter2) - (b1 + b2)/2)
    #
    #         if 0 < distance < 30:
    #             cv2.rectangle(my_img_copy, (panel.x_min, panel.y_min), (panel.x_max, panel.y_max), (255, 255, 0), 10)
    #             cv2.rectangle(my_img_copy, (bbox_near_hit_x_min, bbox_near_hit_y_min), (bbox_near_hit_x_max, bbox_near_hit_y_max), (255, 255, 0), 10)
    #
    #
    #         # distance = rect_distance(panel.x_min, panel.y_min, panel.x_max, panel.y_max,
    #         #                          bbox_near_hit_x_min, bbox_near_hit_y_min, bbox_near_hit_x_max, bbox_near_hit_y_max)
    #         print(distance)
    # hits_near = idx.nearest((0, 0, 10, 10), 3, objects=True)

    # for bbox in my_panels:
    #     if not bbox.is_small:
    #         color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
    #         cv2.rectangle(my_img_copy, (bbox.x_min, bbox.y_min), (bbox.x_max, bbox.y_max), color, 3)

    # # TODO: buscar nearest
    # # TODO: ¿unir, descartar?
    #
    # stack = np.hstack((my_img, my_th))
    # # # stack2 = np.hstack((drawing, result))
    # cv2.namedWindow('finalImg', cv2.WINDOW_NORMAL)
    # cv2.imshow("finalImg", stack)
    # cv2.namedWindow('Proceso color', cv2.WINDOW_NORMAL)
    # cv2.imshow("Proceso color", my_img_copy)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return result, my_img_copy
    #
    # new_filename = os.path.basename(filename)
    # cv2.imwrite(os.path.join('./results/', new_filename), my_img_copy)
    #
    # # TODO: librería previa a la red que trocea la imagen si es muy grande para la entrada de la red


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    root_dir = '/home/garciamorenc/tfm/dataset_comics/fdataset_hr_multiscale/'
    result_dir = '/home/garciamorenc/tfm/dataset_comics/fdataset_hr_multiscale_sub/'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    counter = 0
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            clipped_panels, original_image = run_clip(os.path.join(subdir, file))
            for panel in clipped_panels:
                new_file_name = 'img{:07d}.png'.format(counter)
                new_file_path = os.path.join(result_dir, new_file_name)
                print(new_file_name)
                cropped_image = original_image[panel.y_min:panel.y_max, panel.x_min:panel.x_max]
                # cv2.namedWindow('finalImg', cv2.WINDOW_NORMAL)
                # cv2.imshow("finalImg", cropped_image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                cv2.imwrite(new_file_path, cropped_image)
                counter = counter + 1
