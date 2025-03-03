"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【1012】空间滤波之 Scharr 算子
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv.imread("../images/Fig1001.png", flags=0)

    # (1) 使用函数 filter2D 实现
    kernScharrX = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]])
    kernScharrY = np.array([[-3, 10, -3], [0, 0, 10], [3, 10, 3]])
    ScharrX = cv.filter2D(img, -1, kernScharrX)
    ScharrY = cv.filter2D(img, -1, kernScharrY)

    # (2) 使用 cv.Scharr 实现
    imgScharrX = cv.Scharr(img, cv.CV_32F, 1, 0)  # 水平方向
    imgScharrY = cv.Scharr(img, cv.CV_32F, 0, 1)  # 垂直方向
    absScharrX = cv.convertScaleAbs(imgScharrX)  # 转回 uint8
    absScharrY = cv.convertScaleAbs(imgScharrY)  # 转回 uint8
    ScharrXY = cv.add(absScharrX, absScharrY)  # 用绝对值近似平方根

    plt.figure(figsize=(9, 3.2))
    plt.subplot(131), plt.axis('off'), plt.title("(1) ScharrX(abs)")
    plt.imshow(absScharrX, cmap='gray', vmin=0, vmax=255)
    plt.subplot(132), plt.axis('off'), plt.title("(2) ScharrY(abs)")
    plt.imshow(absScharrY, cmap='gray', vmin=0, vmax=255)
    plt.subplot(133), plt.axis('off'), plt.title("(3) ScharrXY")
    plt.imshow(ScharrXY, cmap='gray', vmin=100, vmax=255)
    plt.tight_layout()
    plt.show()
