import cv2 as cv
import numpy as np
import random
from skimage.util import random_noise
import torch
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop
from basicsr.utils.img_process_util import filter2D
from torch.nn import functional as F

from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels


class DataAugmentation:

    def __init__(self):
        self.blur_kernel_size1 = 21
        self.resize_prob1 = [0.2, 0.7, 0.1]  # up, down, keep
        self.resize_range1 = [0.15, 1.5]
        self.gaussian_noise_prob1 = 0.5
        self.gaussian_noise_range1 = [1, 30]

        self.blur_prob2 = 0.8
        self.blur_kernel_size2 = 21
        self.resize_prob2 = [0.3, 0.4, 0.3]  # up, down, keep
        self.resize_range2 = [0.3, 1.2]
        self.gaussian_noise_prob2 = 0.5
        self.gaussian_noise_range2 = [1, 25]

        self.main_scale = 4
        self.__setup__()

    def __setup__(self):
        self.kernel1 = np.ones((self.blur_kernel_size1, self.blur_kernel_size1), np.float32)\
                       / (self.blur_kernel_size1*self.blur_kernel_size1)
        self.kernel2 = np.ones((self.blur_kernel_size2, self.blur_kernel_size2), np.float32) \
                       / (self.blur_kernel_size2 * self.blur_kernel_size2)

        self.kernel_range = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21
        kernel_size = random.choice(self.kernel_range)
        if kernel_size < 13:
            omega_c = np.random.uniform(np.pi / 3, np.pi)
        else:
            omega_c = np.random.uniform(np.pi / 5, np.pi)
        self.my_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
        print(self.my_kernel.size())

    def feed_data(self, input_img):
        """Accept data from dataloader, and then add two-order degradations to obtain LQ images.
        """
        ori_h, ori_w, _ = input_img.shape
        # ----------------------- The first degradation process ----------------------- #
        # blur

        out = cv.filter2D(src=input_img, ddepth=-1, kernel=self.kernel1)

        # random resize
        updown_type = random.choices(['up', 'down', 'keep'], self.resize_prob1)[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, self.resize_range1[1])
        elif updown_type == 'down':
            scale = np.random.uniform(self.resize_range1[0], 1)
        else:
            scale = 1
        scale_mode = random.choice([cv.INTER_AREA, cv.INTER_LINEAR, cv.INTER_CUBIC])
        scale_width = int(ori_w * scale)
        scale_height = int(ori_h * scale)
        scale_dim = (scale_width, scale_height)
        out = cv.resize(out, scale_dim, interpolation=scale_mode)
        # add noise
        if np.random.uniform() < self.gaussian_noise_prob1:
            gauss_var = random.randrange(self.gaussian_noise_range1[0], self.gaussian_noise_range1[1]) / 100
            noise_img = random_noise(out, mode='gaussian', seed=None, clip=True, mean=0, var=gauss_var)
            out = np.array(255 * noise_img, dtype='uint8')
        else:
            noise_img = random_noise(out, mode='poisson', seed=None, clip=True)
            out = np.array(255 * noise_img, dtype='uint8')
        # # JPEG compression
        # encode_param = [int(cv.IMWRITE_JPEG_QUALITY), 95]
        # result, encimg = cv.imencode('.jpg', out, encode_param)
        # out2 = cv.imdecode(encimg, 1)
        # print('test')


        # ----------------------- The second degradation process ----------------------- #
        # blur
        if np.random.uniform() < self.blur_prob2:
            out = cv.filter2D(src=out, ddepth=-1, kernel=self.kernel2)
        # random resize
        updown_type = random.choices(['up', 'down', 'keep'], self.resize_prob2)[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, self.resize_range2[1])
        elif updown_type == 'down':
            scale = np.random.uniform(self.resize_range2[0], 1)
        else:
            scale = 1
        scale_mode = random.choice([cv.INTER_AREA, cv.INTER_LINEAR, cv.INTER_CUBIC])
        scale_width = int(ori_w / self.main_scale * scale)
        scale_height = int(ori_h / self.main_scale * scale)
        scale_dim = (scale_width, scale_height)
        out = cv.resize(out, scale_dim, interpolation=scale_mode)
        # add noise
        if np.random.uniform() < self.gaussian_noise_prob2:
            gauss_var = random.randrange(self.gaussian_noise_range2[0], self.gaussian_noise_range2[1]) / 100
            noise_img = random_noise(out, mode='gaussian', seed=None, clip=True, mean=0, var=gauss_var)
            out = np.array(255 * noise_img, dtype='uint8')
        else:
            noise_img = random_noise(out, mode='poisson', seed=None, clip=True)
            out = np.array(255 * noise_img, dtype='uint8')

        # JPEG compression + the final sinc filter
        # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
        # as one operation.
        # We consider two orders:
        #   1. [resize back + sinc filter] + JPEG compression
        #   2. JPEG compression + [resize back + sinc filter]
        # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
        scale_mode = random.choice([cv.INTER_AREA, cv.INTER_LINEAR, cv.INTER_CUBIC])
        scale_width = int(ori_w // self.main_scale)
        scale_height = int(ori_h // self.main_scale)
        scale_dim = (scale_width, scale_height)
        out = cv.resize(out, scale_dim, interpolation=scale_mode)
        # TODO filter2D

        # cv.namedWindow('result', cv.WINDOW_NORMAL)
        # cv.imshow('result', out)
        # cv.imwrite('result.png', out)
        # cv.waitKey()
        return out
