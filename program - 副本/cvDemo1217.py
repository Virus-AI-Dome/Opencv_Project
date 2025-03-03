"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【1217】基于灰度形态学的粒度测定
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv.imread("../images/Fig1208.png", flags=0)  # 灰度图像
    _, imgBin = cv.threshold(img, 205, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)  # 二值处理 (黑色背景)
    plt.figure(figsize=(9, 6))
    plt.subplot(231), plt.axis("off"), plt.title("(1) Original")
    plt.imshow(img, cmap='gray')

    # 用不同半径圆形结构元进行开运算
    rList = [14, 21, 28, 35, 43]
    for i in range(5):
        size = rList[i] * 2 + 1
        element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (size, size))  # 圆形结构元
        imgOpen = cv.morphologyEx(imgBin, cv.MORPH_OPEN, element)
        plt.subplot(2, 3, i + 2), plt.title("({}) Opening (r={})".format(i+2, rList[i]))
        plt.imshow(cv.bitwise_not(imgOpen), cmap='gray'), plt.axis("off")
    plt.tight_layout()
    plt.show()

    # 计算圆形直径的半径分布
    maxSize = 43
    sumSurf = np.zeros(maxSize)
    deltaSum = np.zeros(maxSize)
    for r in range(5, maxSize):
        size = r * 2 + 1
        element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (size, size))  # 圆形结构元
        imgOpen = cv.morphologyEx(img, cv.MORPH_OPEN, element)
        sumSurf[r] = np.concatenate(imgOpen).sum()
        deltaSum[r] = sumSurf[r-1] - sumSurf[r]
        print(r, sumSurf[r], deltaSum[r])
    r = range(maxSize)
    plt.figure(figsize=(6, 4.2))
    plt.plot(r[6:], deltaSum[6:], 'b-o')
    plt.title("Delta of surface area")
    plt.xlabel("Diameter"), plt.ylabel("Quantity")
    plt.yticks([])
    plt.show()
