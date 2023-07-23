import cv2 as cv
import numpy as np
import random

from basicsr import DiffJPEG
from skimage.util import random_noise
import torch
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop
from basicsr.utils.img_process_util import filter2D
from torch.nn import functional as F

import torchvision.transforms as transforms
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
import math
from basicsr import USMSharp


class DataAugmentation2:

    def __init__(self):
        self.scale = 2
        self.kernel_range = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21
        self.jpeger = DiffJPEG(differentiable=False)  # simulate JPEG compression artifacts
        self.pulse_tensor = np.zeros((21, 21), np.float32)  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1

        # the first degradation process
        self.sinc_prob = 0.1
        self.kernel_list = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
        self.kernel_prob = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
        self.blur_sigma = [0.2, 3]
        self.betag_range = [0.5, 4]
        self.betap_range = [1, 2]

        self.sinc_prob2 = 0.1
        self.kernel_list2 = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
        self.kernel_prob2 = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
        self.blur_sigma2 = [0.2, 1.5]
        self.betag_range2 = [0.5, 4]
        self.betap_range2 = [1, 2]

        self.final_sinc_prob = 0.8

        self.resize_prob = [0.2, 0.7, 0.1]  # up, down, keep
        self.resize_range = [0.15, 1.5]
        self.gray_noise_prob = 0.4
        self.gaussian_noise_prob = 0.5
        self.noise_range = [1, 30]
        self.poisson_scale_range = [0.05, 3]
        self.jpeg_range = [30, 95]

        # the second degradation process
        self.second_blur_prob = 0.8
        self.resize_prob2 = [0.3, 0.4, 0.3]  # up, down, keep
        self.resize_range2 = [0.3, 1.2]
        self.gray_noise_prob2 = 0.4
        self.gaussian_noise_prob2 = 0.5
        self.noise_range2 = [1, 25]
        self.poisson_scale_range2 = [0.05, 2.5]
        self.jpeg_range2 = [30, 95]

        ################################################################################################################
        self.__setup__()

    def __setup__(self):
        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.sinc_prob:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma, [-math.pi, math.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None)
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        self.kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.sinc_prob2:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2, [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None)

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        self.kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------- the final sinc kernel ------------------------------------- #
        if np.random.uniform() < self.final_sinc_prob:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            self.sinc_kernel = torch.FloatTensor(sinc_kernel)
            self.sinc_kernel = self.sinc_kernel.squeeze_(0)
            self.sinc_kernel = self.sinc_kernel.detach().numpy()
        else:
            self.sinc_kernel = self.pulse_tensor

    def feed_data(self, input_img):
        """Accept data from dataloader, and then add two-order degradations to obtain LQ images.
        """
        ori_h, ori_w, _ = input_img.shape
        # ----------------------- The first degradation process ----------------------- #
        # blur
        out = cv.filter2D(src=input_img, ddepth=-1, kernel=self.kernel)
        # random resize
        updown_type = random.choices(['up', 'down', 'keep'], self.resize_prob)[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, self.resize_range[1])
        elif updown_type == 'down':
            scale = np.random.uniform(self.resize_range[0], 1)
        else:
            scale = 1
        scale_mode = random.choice([cv.INTER_AREA, cv.INTER_LINEAR, cv.INTER_CUBIC])
        scale_width = int(ori_w * scale)
        scale_height = int(ori_h * scale)
        scale_dim = (scale_width, scale_height)
        out = cv.resize(out, scale_dim, interpolation=scale_mode)
        # add noise
        # cv2 to tensor
        gray_noise_prob = self.gray_noise_prob
        transform = transforms.ToTensor()
        out = transform(out).unsqueeze_(0)
        if np.random.uniform() < self.gaussian_noise_prob:
            out = random_add_gaussian_noise_pt(
                out, sigma_range=self.noise_range, clip=True, rounds=False, gray_prob=gray_noise_prob)
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=self.poisson_scale_range,
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False)
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.jpeg_range)
        out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
        out = self.jpeger(out, quality=jpeg_p)

        # tensor to cv2
        out = out.squeeze_(0)
        numpy_image = out.detach().numpy()
        out = np.transpose(numpy_image, (1, 2, 0))

        # ----------------------- The second degradation process ----------------------- #
        # blur
        if np.random.uniform() < self.second_blur_prob:
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
        scale_width = int(ori_w / self.scale * scale)
        scale_height = int(ori_h / self.scale * scale)
        scale_dim = (scale_width, scale_height)
        out = cv.resize(out, scale_dim, interpolation=scale_mode)
        # add noise
        # cv2 to tensor
        gray_noise_prob = self.gray_noise_prob2
        transform = transforms.ToTensor()
        out = transform(out).unsqueeze_(0)
        if np.random.uniform() < self.gaussian_noise_prob2:
            out = random_add_gaussian_noise_pt(
                out, sigma_range=self.noise_range2, clip=True, rounds=False, gray_prob=gray_noise_prob)
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=self.poisson_scale_range2,
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False)

        # JPEG compression + the final sinc filter
        # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
        # as one operation.
        # We consider two orders:
        #   1. [resize back + sinc filter] + JPEG compression
        #   2. JPEG compression + [resize back + sinc filter]
        # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
        if np.random.uniform() < 0.5:
            # tensor to cv2
            out = out.squeeze_(0)
            numpy_image = out.detach().numpy()
            out = np.transpose(numpy_image, (1, 2, 0))
            # resize back + the final sinc filter
            scale_mode = random.choice([cv.INTER_AREA, cv.INTER_LINEAR, cv.INTER_CUBIC])
            scale_width = int(ori_w // self.scale)
            scale_height = int(ori_h // self.scale)
            scale_dim = (scale_width, scale_height)
            out = cv.resize(out, scale_dim, interpolation=scale_mode)
            out = cv.filter2D(src=out, ddepth=-1, kernel=self.sinc_kernel)
            # cv2 to tensor
            transform = transforms.ToTensor()
            out = transform(out).unsqueeze_(0)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.jpeg_range2)
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
            # tensor to cv2
            out = out.squeeze_(0)
            numpy_image = out.detach().numpy()
            out = np.transpose(numpy_image, (1, 2, 0))
        else:
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.jpeg_range2)
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
            # tensor to cv2
            out = out.squeeze_(0)
            numpy_image = out.detach().numpy()
            out = np.transpose(numpy_image, (1, 2, 0))
            # resize back + the final sinc filter
            scale_mode = random.choice([cv.INTER_AREA, cv.INTER_LINEAR, cv.INTER_CUBIC])
            scale_width = int(ori_w // self.scale)
            scale_height = int(ori_h // self.scale)
            scale_dim = (scale_width, scale_height)
            out = cv.resize(out, scale_dim, interpolation=scale_mode)
            out = cv.filter2D(src=out, ddepth=-1, kernel=self.sinc_kernel)

        # cv2 to tensor
        transform = transforms.ToTensor()
        out = transform(out).unsqueeze_(0)
        # clamp and round
        out = torch.clamp((out * 255.0).round(), 0, 255) / 255.
        usm_sharpener = USMSharp()
        out = usm_sharpener(out)

        # tensor to cv2
        out = out.squeeze_(0)
        numpy_image = out.detach().numpy()
        out = np.transpose(numpy_image, (1, 2, 0))

        # cv.namedWindow('result', cv.WINDOW_NORMAL)
        # cv.imshow('result', out)
        # # cv.imwrite('result.png', out)
        # cv.waitKey()
        return 255*out
