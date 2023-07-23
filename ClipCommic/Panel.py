import numpy as np


class Panel:
    DEFAULT_SMALL_RATIO = 1 / 100
    DEFAULT_MEDIUM_RATIO = 1 / 10
    size_classification = None

    def __init__(self, index_intersections, bbox_intersections):
        self.index_intersections = index_intersections
        min_coords = np.array(bbox_intersections).min(axis=0)
        max_coords = np.array(bbox_intersections).max(axis=0)
        self.x_min = min_coords[0]
        self.y_min = min_coords[1]
        self.x_max = max_coords[2]
        self.y_max = max_coords[3]

    def __add__(self, index_intersections, bbox_intersections):
        self.index_intersections = list(set(index_intersections + self.index_intersections))
        aux = [np.array([self.x_min, self.y_min, self.x_max, self.y_max])] + bbox_intersections
        min_coords = np.array(aux).min(axis=0)
        max_coords = np.array(aux).max(axis=0)
        self.x_min = min_coords[0]
        self.y_min = min_coords[1]
        self.x_max = max_coords[2]
        self.y_max = max_coords[3]

    # def __is_small__(self, image_size):
    #     self.is_small = (((self.x_max - self.x_min) < image_size[0] * self.DEFAULT_SMALL_RATIO)
    #                      and ((self.y_max - self.y_min) < image_size[1] * self.DEFAULT_SMALL_RATIO))
    #     return self.is_small

    def __size_classifier__(self, image_size):
        is_small = (((self.x_max - self.x_min) < image_size[0] * self.DEFAULT_SMALL_RATIO)
                    and ((self.y_max - self.y_min) < image_size[1] * self.DEFAULT_SMALL_RATIO))

        is_medium = (((self.x_max - self.x_min) < image_size[0] * self.DEFAULT_MEDIUM_RATIO)
                    and ((self.y_max - self.y_min) < image_size[1] * self.DEFAULT_MEDIUM_RATIO))

        if is_small:
            self.size_classification = 'small'
        elif is_medium:
            self.size_classification = 'medium'
        else:
            self.size_classification = 'big'

        return self.size_classification
