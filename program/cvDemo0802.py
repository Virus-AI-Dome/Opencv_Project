"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【0802】灰度图像的直方图均衡
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    gray = cv.imread("../images/Fig0702.png", flags=0)  # flags=0 读取为灰度图像
    # gray = cv.multiply(gray, 0.6)  # 调整原始图像，用于比较归一化的作用

    # 直方图均衡化
    histSrc = cv.calcHist([gray], [0], None, [256], [0, 256])
    grayEqualize = cv.equalizeHist(gray)
    histEqual = cv.calcHist([grayEqualize], [0], None, [256], [0, 256])

    # 直方图归一化
    grayNorm = cv.normalize(gray, None, 0, 255, cv.NORM_MINMAX)
    histNorm = cv.calcHist([grayNorm], [0], None, [256], [0, 256])

    plt.figure(figsize=(9, 6))
    plt.subplot(231), plt.axis('off'), plt.title("(1) Original")
    plt.imshow(gray, cmap='gray', vmin=0, vmax=255)  # 原始图像
    plt.subplot(232), plt.axis('off'), plt.title("(2) Normalized")
    plt.imshow(grayNorm, cmap='gray', vmin=0, vmax=255)  # 原始图像
    plt.subplot(233), plt.axis('off'), plt.title("(3) Hist-equalized")
    plt.imshow(grayEqualize, cmap='gray', vmin=0, vmax=255) # 转换图像
    plt.subplot(234, yticks=[]), plt.axis([0, 255, 0, np.max(histSrc)])
    plt.title("(4) Gray hist of src")
    plt.bar(range(256), histSrc[:, 0])  # 原始直方图
    plt.subplot(235, yticks=[]), plt.axis([0, 255, 0, np.max(histSrc)])
    plt.title("(5) Gray hist of normalized")
    plt.bar(range(256), histNorm[:, 0])  # 原始直方图
    plt.subplot(236, yticks=[]), plt.axis([0, 255, 0, np.max(histSrc)])
    plt.title("(6) Gray histm of equalized")
    plt.bar(range(256), histEqual[:, 0])  # 均衡直方图
    plt.tight_layout()
    plt.show()

