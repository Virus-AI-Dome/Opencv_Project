"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【1209】形态学算法之边界提取
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv.imread("../images/Fig0801.png", flags=0)  # 读取灰度图像
    _, imgBin = cv.threshold(img, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)  # 二值处理

    # 3*3 矩形结构元
    element = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    imgErode1 = cv.erode(imgBin, kernel=element)  # 图像腐蚀
    imgBound1 = imgBin - imgErode1  # 图像边界提取
    # 5*5 矩形结构元
    element = cv.getStructuringElement(cv.MORPH_RECT, (9, 9))
    imgErode2 = cv.erode(imgBin, kernel=element)  # 图像腐蚀
    imgBound2 = imgBin - imgErode2  # 图像边界提取

    plt.figure(figsize=(9, 3.3))
    plt.subplot(131), plt.axis('off'), plt.title("(1) Original")
    plt.imshow(imgBin, cmap='gray', vmin=0, vmax=255)
    plt.subplot(132), plt.title("(2) Boundary extraction (3,3)"), plt.axis('off')
    plt.imshow(imgBound1, cmap='gray', vmin=0, vmax=255)
    plt.subplot(133), plt.title("(3) Boundary extraction (9,9)"), plt.axis('off')
    plt.imshow(imgBound2, cmap='gray', vmin=0, vmax=255)
    plt.tight_layout()
    plt.show()

