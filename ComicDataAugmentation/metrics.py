from math import log10, sqrt
import cv2
import numpy as np
import os
import statistics
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from basicsr.metrics import niqe

# import imutils

def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if (mse == 0):  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    hr_dir = '/home/garciamorenc/tfm/Real-ESRGAN/datasets/fdataset_hr_test_600/'
    prediction_dir = '/home/garciamorenc/tfm/Real-ESRGAN/results_x2_100k_5e-5lr_3batch//'
    list_psnr = []
    list_ssim = []
    list_psnr2 = []
    list_niqe = []

    for filename in os.listdir(hr_dir):
        original_path = os.path.join(hr_dir, filename)
        # filename_out = filename.replace('.png', '_out.png')
        compressed_path = os.path.join(prediction_dir, filename)

        original = cv2.imread(original_path)
        compressed = cv2.imread(compressed_path, 1)

        ori_h, ori_w, _ = original.shape
        comp_h, comp_w, _ = compressed.shape
        if ori_h != comp_h:
            scale_dim = (ori_h, ori_h)
            original = cv2.resize(original, scale_dim, interpolation=cv2.INTER_AREA)
            compressed = cv2.resize(compressed, scale_dim, interpolation=cv2.INTER_AREA)

        value = PSNR(original, compressed)
        list_psnr.append(value)
        print(f"PSNR value is {value} dB")
        value2 = peak_signal_noise_ratio(original, compressed)
        list_psnr2.append(value2)
        print(f"PSNR value is {value2} dB")

        # 4. Convert the images to grayscale
        grayA = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(compressed, cv2.COLOR_BGR2GRAY)

        # 5. Compute the Structural Similarity Index (SSIM) between the two
        #    images, ensuring that the difference image is returned
        (score, diff) = structural_similarity(grayA, grayB, full=True)
        diff = (diff * 255).astype("uint8")

        # 6. You can print only the score if you want
        print(f"SSIM: score: {score}")
        list_ssim.append(score)


        # --------------------------------------
        niqe_result = niqe.calculate_niqe(compressed, 2)
        print(f"NIQE: score: {niqe_result}")
        list_niqe.append(niqe_result)


    median_pnsr = statistics.median(list_psnr)
    print(f"PSNR median value is {median_pnsr} dB")
    median_pnsr2 = statistics.median(list_psnr2)
    print(f"PSNR2 median value is {median_pnsr2} dB")

    median_pnsr = statistics.median(list_ssim)
    print(f"SSIM median value is {median_pnsr}")

    median_niqe = statistics.median(list_niqe)
    print(f"NIQE median value is {median_niqe}")
