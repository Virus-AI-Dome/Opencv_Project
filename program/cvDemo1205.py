"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【1205】灰度级形态学运算的原理图
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    lens = 640
    t = np.arange(1, lens + 1)  # start, end, step
    y = 2 * np.sin(0.01 * t) + np.sin(0.02 * t) + np.sin(0.05 * t) + np.pi
    y2D = np.reshape(y, (-1,1))  # (640,1)
    img = np.uint8(cv.normalize(y2D, None, 0, 255, cv.NORM_MINMAX))

    lenSE = 50
    element = cv.getStructuringElement(cv.MORPH_RECT, (1, lenSE))  # 条形结构元
    imgErode = cv.erode(img, element)  # 灰度腐蚀
    imgDilate = cv.dilate(img, element)  # 灰度膨胀
    imgOpen = cv.morphologyEx(img, cv.MORPH_OPEN, element)  # 灰度开运算
    imgClose = cv.morphologyEx(img, cv.MORPH_CLOSE, element)  # 灰度闭运算
    imgThat = cv.morphologyEx(img, cv.MORPH_TOPHAT, element)  # 顶帽运算
    imgBhat = cv.morphologyEx(img, cv.MORPH_BLACKHAT, element)  # 底帽运算

    print(t.shape, y.shape, img.shape, element.shape)
    print(img.max(), img.min())
    plt.figure(figsize=(9, 6))
    plt.subplot(231), plt.xticks([]), plt.yticks([])
    plt.title("(1) Gray erosion profile")
    plt.plot(img, 'k--', imgErode, 'b-')  # 灰度腐蚀
    plt.subplot(232), plt.xticks([]), plt.yticks([])
    plt.title("(2) Gray opening profile")
    plt.plot(img, 'k--', imgOpen, 'b-')  # 灰度开运算
    plt.subplot(233), plt.xticks([]), plt.yticks([])
    plt.title("(3) Gray tophat profile")
    plt.plot(img, 'k--', imgThat, 'b-')  # 灰度顶帽运算
    plt.subplot(234), plt.xticks([]), plt.yticks([])
    plt.title("(4) Gray dilation profile")
    plt.plot(img, 'k--', imgDilate, 'b-')  # 灰度膨胀
    plt.subplot(235), plt.xticks([]), plt.yticks([])
    plt.title("(5) Gray closing profile")
    plt.plot(img, 'k--', imgClose, 'b-')  # 灰度闭运算
    plt.subplot(236), plt.xticks([]), plt.yticks([])
    plt.title("(6) Gray blackhat profile")
    plt.plot(img, 'k--', imgBhat, 'b-')  # 灰度底帽运算
    plt.tight_layout()
    plt.show()
