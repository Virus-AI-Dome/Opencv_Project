"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【0801】灰度图像与彩色图像的直方图
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv.imread("../images/Lena.tif", flags=1)  # 读取彩色图像
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 转为灰度图像

    # OpenCV：cv.calcHist 计算灰度直方图
    histCV = cv.calcHist([gray], [0], None, [256], [0, 255])  # (256,1)

    # Numpy：np.histogram 计算灰度直方图
    histNP, bins = np.histogram(gray.flatten(), 256)  # (256,)

    print(histCV.shape, histNP.shape)
    print(histCV.max(), histNP. max())
    plt.figure(figsize=(9, 3))
    plt.subplot(131, yticks=[]), plt.axis([0, 255, 0, np.max(histCV)])
    plt.title("(1) Gray histogram (np.histogram)")
    plt.bar(bins[:-1], histNP)
    plt.subplot(132, yticks=[]), plt.axis([0, 255, 0, np.max(histCV)])
    plt.title("(2) Gray histogram (cv.calcHist)")
    plt.bar(range(256), histCV[:, 0])

    # 计算和绘制彩色图像各通道的直方图
    plt.subplot(133, yticks=[])
    plt.title("(3) Color histograms (cv.calcHist)")
    color = ['b', 'g', 'r']
    for ch, col in enumerate(color):
        histCh = cv.calcHist([img], [ch], None, [256], [0, 255])
        plt.plot(histCh, color=col)
        plt.xlim([0, 256])
    plt.tight_layout()
    plt.show()



