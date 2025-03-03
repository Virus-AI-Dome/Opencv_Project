"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【1206】灰度级形态学运算
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv.imread("../images/Fig1101.png", flags=0)  # 灰度图像

    element = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))  # 矩形结构元
    imgErode = cv.erode(img, element)  # 灰度腐蚀
    imgDilate = cv.dilate(img, element)  # 灰度膨胀
    imgOpen = cv.morphologyEx(img, cv.MORPH_OPEN, element)  # 灰度开运算
    imgClose = cv.morphologyEx(img, cv.MORPH_CLOSE, element)  # 灰度闭运算
    imgGrad = cv.morphologyEx(img, cv.MORPH_GRADIENT, element)  # 灰度级梯度

    plt.figure(figsize=(9, 6))
    plt.subplot(231), plt.axis('off'), plt.title("(1) Original")
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.subplot(232), plt.title("(2) Grayscale erosion"), plt.axis('off')
    plt.imshow(imgErode, cmap='gray', vmin=0, vmax=255)
    plt.subplot(233), plt.title("(3) Grayscale dilation"), plt.axis('off')
    plt.imshow(imgDilate, cmap='gray', vmin=0, vmax=255)
    plt.subplot(234), plt.title("(4) Grayscale opening"), plt.axis('off')
    plt.imshow(imgOpen, cmap='gray', vmin=0, vmax=255)
    plt.subplot(235), plt.title("(5) Grayscale closing"), plt.axis('off')
    plt.imshow(imgClose, cmap='gray', vmin=0, vmax=255)
    plt.subplot(236), plt.title("(6) Grayscale gradient"), plt.axis('off')
    plt.imshow(imgGrad, cmap='gray', vmin=0, vmax=255)
    plt.tight_layout()
    plt.show()
