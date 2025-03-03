"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【1008】空间滤波之双边滤波器
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv.imread("../images/LenaGauss.png", flags=0)  # 读取灰度图像
    hImg, wImg = img.shape[:2]
    print(hImg, wImg)

    # (1) 高斯低通滤波核
    ksize = (11, 11)  # 高斯滤波器尺寸
    imgGaussianF = cv.GaussianBlur(img, ksize, 0, 0)
    # (2) 双边滤波器
    imgBilateralF = cv.bilateralFilter(img, d=5, sigmaColor=40, sigmaSpace=10)

    plt.figure(figsize=(9, 3.5))
    plt.subplot(131), plt.axis('off'), plt.title("(1) Original")
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.subplot(132), plt.axis('off'), plt.title("(2) GaussianFilter")
    plt.imshow(imgGaussianF, cmap='gray', vmin=0, vmax=255)
    plt.subplot(133), plt.axis('off'), plt.title("(3) BilateralFilter")
    plt.imshow(imgBilateralF, cmap='gray', vmin=0, vmax=255)
    plt.tight_layout()
    plt.show()

