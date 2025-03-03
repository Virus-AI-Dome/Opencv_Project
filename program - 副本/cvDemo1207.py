"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【1207】灰度顶帽变换校正光照影响
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv.imread("../images/Fig1203.png", flags=0)  # 灰度图像

    # 直接用 Otsu 最优阈值处理方法进行二值化处理
    _, imgBin1 = cv.threshold(img, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)  # 二值处理

    # 顶帽运算后再用 Otsu 最优阈值处理方法进行二值化处理
    element = cv.getStructuringElement(cv.MORPH_RECT, (60, 80))  # 矩形结构元
    imgThat = cv.morphologyEx(img, cv.MORPH_TOPHAT, element)  # 顶帽运算
    ret, imgBin2 = cv.threshold(imgThat, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)  # 二值处理

    fig = plt.figure(figsize=(9, 6))
    plt.subplot(231), plt.title("(1) Original"), plt.axis('off')
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.subplot(234), plt.title("(4) Tophat"), plt.axis('off')
    plt.imshow(imgThat, cmap='gray', vmin=0, vmax=255)
    plt.subplot(233), plt.title("(3) Original binary"), plt.axis('off')
    plt.imshow(imgBin1, cmap='gray', vmin=0, vmax=255)
    plt.subplot(236), plt.title("(6) Tophat binary"), plt.axis('off')
    plt.imshow(imgBin2, cmap='gray', vmin=0, vmax=255)
    h = np.arange(0, img.shape[1])
    w = np.arange(0, img.shape[0])
    xx, yy = np.meshgrid(h, w)  # 转换为网格点集（二维数组）
    ax1 = plt.subplot(232, projection='3d')
    ax1.plot_surface(xx, yy, img, cmap='coolwarm')
    # ax1.plot_surface(xx, yy, img, cmap='gray')
    ax1.set_xticks([]), ax1.set_yticks([]), ax1.set_zticks([])
    ax1.set_title("(2) Original grayscale")
    ax2 = plt.subplot(235, projection='3d')
    ax2.plot_surface(xx, yy, imgThat, cmap='coolwarm')
    # ax2.plot_surface(xx, yy, imgThat, cmap='gray')
    ax2.set_xticks([]), ax2.set_yticks([]), ax2.set_zticks([])
    ax2.set_title("(5) Tophat grayscale")
    plt.tight_layout()
    plt.show()
