"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【0704】灰度变换之对数变换
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    gray = cv.imread("../images/Fig0602.png", flags=0)  # 读取为灰度图像

    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    amp = np.abs(fft_shift)
    ampNorm = np.uint8(cv.normalize(amp, None, 0, 255, cv.NORM_MINMAX))
    ampLog =np.abs(np.log(1.0 + np.abs(fft_shift)))
    ampLogNorm = np.uint8(cv.normalize(ampLog, None, 0, 255, cv.NORM_MINMAX))

    plt.figure(figsize=(9, 3.2))
    plt.subplot(131), plt.title("(1) Original"), plt.axis('off')
    plt.imshow(gray, cmap='gray', vmin=0, vmax=255)
    plt.subplot(132), plt.title("(2) FFT spectrum"), plt.axis('off')
    plt.imshow(ampNorm, cmap='gray', vmin=0, vmax=255)
    plt.subplot(133), plt.title("(3) LogTrans of FFT"), plt.axis('off')
    plt.imshow(ampLogNorm, cmap='gray', vmin=0, vmax=255)
    plt.tight_layout()
    plt.show()

