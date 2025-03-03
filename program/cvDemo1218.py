"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【1218】基于形态学的角点检测

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    # 基于灰度形态学的复杂背景图像重建
    img = cv.imread("../images/Fig1209.png", flags=1)
    imgSign = img.copy()
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # 边缘检测
    element = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    imgEdge = cv.morphologyEx(imgGray, cv.MORPH_GRADIENT, element)  # 形态学梯度

    # 构造 9×9 结构元素，十字形、菱形、方形、X 型
    cross = cv.getStructuringElement(cv.MORPH_CROSS, (9, 9))  # 十字型结构元
    square = cv.getStructuringElement(cv.MORPH_RECT, (9, 9))  # 矩形结构元
    xShape = cv.getStructuringElement(cv.MORPH_CROSS, (9, 9))  # X 形结构元
    diamond = cv.getStructuringElement(cv.MORPH_CROSS, (9, 9))  # 构造菱形结构元
    diamond[1, 1] = diamond[3, 3] = 1
    diamond[1, 3] = diamond[3, 1] = 1
    print(diamond)

    imgDilate1 = cv.dilate(imgGray, cross)  # 用十字型结构元膨胀原图像
    imgErode1 = cv.erode(imgDilate1, diamond)  # 用菱形结构元腐蚀图像

    imgDilate2 = cv.dilate(imgGray, xShape)  # 使用 X 形结构元膨胀原图像
    imgErode2 = cv.erode(imgDilate2, square)  # 使用方形结构元腐蚀图像

    imgDiff = cv.absdiff(imgErode2, imgErode1)  # 将两幅闭运算的图像相减获得角点
    retval, thresh = cv.threshold(imgDiff, 40, 255, cv.THRESH_BINARY)  # # 二值化处理

    # 在原图上用十字标记角点
    for j in range(thresh.size):
        yc = int(j / thresh.shape[0])
        xc = int(j % thresh.shape[0])
        if (thresh[xc,yc] == 255):
            cv.drawMarker(imgSign, (yc, xc), (0,0,255), cv.MARKER_CROSS, 10)  # 在点(x,y)标记

    plt.figure(figsize=(9, 3.5))
    plt.subplot(131), plt.title("(1) Original"), plt.axis('off')
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.subplot(132), plt.title("(2) Morph edge"), plt.axis('off')
    plt.imshow(cv.bitwise_not(imgEdge), cmap='gray', vmin=0, vmax=255)
    plt.subplot(133), plt.title("(3) Morph corner"), plt.axis('off')
    plt.imshow(cv.cvtColor(imgSign, cv.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()
