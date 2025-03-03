"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【1203】形态学运算之形态学梯度
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv.imread("../images/Fig0703.png", flags=0)  # 读取灰度图像
    _, imgBin = cv.threshold(img, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)  # 二值处理

    # 图像的形态学梯度
    element = cv.getStructuringElement(cv.MORPH_RECT, (3,3))  # 矩形结构元
    imgGrad = cv.morphologyEx(imgBin, cv.MORPH_GRADIENT, kernel=element)  # 形态学梯度

    # 开运算 -> 形态学梯度
    imgOpen = cv.morphologyEx(imgBin, cv.MORPH_OPEN, kernel=element)  # 开运算
    imgOpenGrad = cv.morphologyEx(imgOpen, cv.MORPH_GRADIENT, kernel=element)  # 形态学梯度

    plt.figure(figsize=(9, 3.5))
    plt.subplot(131), plt.axis('off'), plt.title("(1) Original")
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.subplot(132), plt.title("(2) MORPH_GRADIENT"), plt.axis('off')
    plt.imshow(imgGrad, cmap='gray', vmin=0, vmax=255)
    plt.subplot(133), plt.title("(3) Opening -> Gradient"), plt.axis('off')
    plt.imshow(imgOpenGrad, cmap='gray', vmin=0, vmax=255)
    plt.tight_layout()
    plt.show()
