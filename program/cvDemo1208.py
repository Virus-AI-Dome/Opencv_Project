"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【1208】灰度底帽变换校正光照影响
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv.imread("../images/Fig1204.png", flags=0)  # 灰度图像
    _, imgBin1 = cv.threshold(img, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)  # 二值处理

    # 底帽运算
    r = 80  # 特征尺寸，由目标的大小确定
    element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (r, r))  # 圆形结构元
    imgBhat = cv.morphologyEx(img, cv.MORPH_BLACKHAT, element)  # 底帽运算
    _, imgBin2 = cv.threshold(imgBhat, 20, 255, cv.THRESH_BINARY)  # 二值处理
    # 闭操作去除圆环的噪点
    element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (9, 9))  # 圆形结构元
    imgSegment = cv.morphologyEx(imgBin2, cv.MORPH_CLOSE, element)  # 闭运算

    fig = plt.figure(figsize=(9, 6))
    plt.subplot(231), plt.title("(1) Original"), plt.axis('off')
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.subplot(234), plt.title("(4) Blackhat"), plt.axis('off')
    plt.imshow(imgBhat, cmap='gray', vmin=0, vmax=255)
    plt.subplot(233), plt.title("(3) Original binary"), plt.axis('off')
    plt.imshow(imgBin1, cmap='gray', vmin=0, vmax=255)
    plt.subplot(236), plt.title("(6) Blackhat binary"), plt.axis('off')
    plt.imshow(imgSegment, cmap='gray', vmin=0, vmax=255)
    h = np.arange(0, img.shape[1])
    w = np.arange(0, img.shape[0])
    xx, yy = np.meshgrid(h, w)  # 转换为网格点集（二维数组）
    ax1 = plt.subplot(232, projection='3d')
    ax1.plot_surface(xx, yy, img, cmap='coolwarm')
    # ax1.plot_surface(xx, yy, img, cmap='gray')
    ax1.set_xticks([]), ax1.set_yticks([]), ax1.set_zticks([])
    ax1.set_title("(2) Original grayscale")
    ax2 = plt.subplot(235, projection='3d')
    ax2.plot_surface(xx, yy, imgBhat, cmap='coolwarm')
    # ax2.plot_surface(xx, yy, imgBhat, cmap='gray')
    ax2.set_xticks([]), ax2.set_yticks([]), ax2.set_zticks([])
    ax2.set_title("(5) Blackhat grayscale")
    plt.tight_layout()
    plt.show()
