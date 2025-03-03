"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【1003】空间滤波之高斯低通滤波器
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv.imread("../images/Fig1001.png", flags=0)  # 读取灰度图像

    # (1) 计算高斯核
    kernX = cv.getGaussianKernel(5, 0)  # 一维高斯核
    kernel = kernX * kernX.T  # 二维高斯核
    print("1D kernel of Gaussian:{}".format(kernX.shape))
    print(kernX.T.round(4))
    print("2D kernel of Gaussian:{}".format(kernel.shape))
    print(kernel.round(4))

    # (2) 高斯低通滤波核
    ksize = (11, 11)  # 高斯滤波器核的尺寸
    GaussBlur11 = cv.GaussianBlur(img, ksize, 0)  # sigma 由 ksize 计算
    ksize = (43, 43)
    GaussBlur43 = cv.GaussianBlur(img, ksize, 0)

    plt.figure(figsize=(9, 3.2))
    plt.subplot(131), plt.axis('off'), plt.title("(1) Original")
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.subplot(132), plt.axis('off'), plt.title("(2) GaussianFilter (k=11)")
    plt.imshow(GaussBlur11, cmap='gray', vmin=0, vmax=255)
    plt.subplot(133), plt.axis('off'), plt.title("(3) GaussianFilter (k=43)")
    plt.imshow(GaussBlur43, cmap='gray', vmin=0, vmax=255)
    plt.tight_layout()
    plt.show()