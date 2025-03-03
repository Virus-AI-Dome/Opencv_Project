"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【1204】击中击不中变换进行特征识别
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv.imread("../images/Fig1202.png", flags=0)  # 读取灰度图像
    _, binary = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)  # 二值处理
    kern = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))  # 圆形结构元
    imgBin = cv.morphologyEx(binary, cv.MORPH_CLOSE, kern)  # 封闭孔洞

    # 击中击不中变换
    kernB1 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (12, 12))
    imgHMT1 = cv.morphologyEx(imgBin, cv.MORPH_HITMISS, kernB1)
    kernB2 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (20, 20))
    imgHMT2 = cv.morphologyEx(imgBin, cv.MORPH_HITMISS, kernB2)

    plt.figure(figsize=(9, 3.3))
    plt.subplot(131), plt.axis('off'), plt.title("(1) Original")
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.subplot(132), plt.title("(2) HITMISS (12,12)"), plt.axis('off')
    plt.imshow(cv.bitwise_not(imgHMT1), cmap='gray', vmin=0, vmax=255)
    plt.subplot(133), plt.title("(3) HITMISS (20,20)"), plt.axis('off')
    plt.imshow(cv.bitwise_not(imgHMT2), cmap='gray', vmin=0, vmax=255)
    plt.tight_layout()
    plt.show()
