"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【0803】灰度图像的直方图匹配
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    graySrc = cv.imread("../images/Fig0702.png", flags=0)  # 读取待匹配的图像
    grayRef = cv.imread("../images/Fig0701.png", flags=0)  # 读取模板图像

    # 计算累计直方图
    histSrc = cv.calcHist([graySrc], [0], None, [256], [0, 256])
    histRef = cv.calcHist([grayRef], [0], None, [256], [0, 256])
    cdfSrc = np.cumsum(histSrc)   # 原始图像累积分布函数 CDF
    cdfRef = np.cumsum(histRef)   # 匹配模板累积分布函数 CDF
    cdfSrc = cdfSrc / cdfSrc[-1]  # 归一化: 0~1
    cdfRef = cdfRef / cdfRef[-1]


    luTable = np.interp(cdfSrc, cdfRef, np.arange(256)).astype(np.uint8)   #找到源图像的 CDF 在目标图像的 CDF 中对应的值：
    grayDst = cv.LUT(graySrc, luTable)

    plt.figure(figsize=(9, 6))
    plt.subplot(231), plt.title("(1) Original"), plt.axis('off')
    plt.imshow(graySrc, cmap='gray')  # 原始图像
    plt.subplot(232), plt.title("(2) Matching template"), plt.axis('off')
    plt.imshow(grayRef, cmap='gray')  # 匹配模板图像
    plt.subplot(233), plt.title("(3) Matched result"), plt.axis('off')
    plt.imshow(grayDst, cmap='gray')  # 匹配结果图像
    plt.subplot(234, xticks=[], yticks=[])
    histSrc = cv.calcHist([graySrc], [0], None, [256], [0, 255])
    plt.title("(4) Original hist")
    plt.bar(range(256), histSrc[:, 0])
    plt.axis([0, 255, 0, np.max(histSrc)])
    plt.subplot(235, xticks=[], yticks=[])
    histRef = cv.calcHist([grayRef], [0], None, [256], [0, 255])
    plt.title("(5) Template hist")
    plt.bar(range(256), histRef[:, 0])
    plt.axis([0, 255, 0, np.max(histRef)])
    plt.subplot(236, xticks=[], yticks=[])
    histDst = cv.calcHist([grayDst], [0], None, [256], [0, 255])
    plt.title("(6) Matched hist")
    plt.bar(range(256), histDst[:, 0])
    plt.axis([0, 255, 0, np.max(histDst)])
    plt.tight_layout()
    plt.show()
