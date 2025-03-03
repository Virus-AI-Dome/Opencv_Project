"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【0905】基于局部性质的自适应阈值处理
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    # img = cv.imread("../images/grayShade2.png", flags=0)
    img = cv.imread("../images/Fig0701.png", flags=0)

    # 自适应局部阈值处理
    binaryMean = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 3, 3)
    binaryGauss = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 3, 3)

    # 参考方法：自适应局部阈值处理
    ratio = 0.03
    imgBlur = cv.boxFilter(img, -1, (3, 3))  # 盒式滤波器，均值平滑
    localThresh = img - (1.0 - ratio) * imgBlur
    binaryBox = np.ones_like(img) * 255  # 创建与 img 相同形状的白色图像
    binaryBox[localThresh < 0] = 0

    plt.figure(figsize=(9, 3))
    plt.subplot(131), plt.axis('off'), plt.title("(1) Adaptive mean")
    plt.imshow(binaryMean, 'gray')
    plt.subplot(132), plt.axis('off'), plt.title("(2) Adaptive Gauss")
    plt.imshow(binaryGauss, 'gray')
    plt.subplot(133), plt.axis('off'), plt.title("(3) Adaptive local thresh")
    plt.imshow(binaryBox, 'gray')
    plt.tight_layout()
    plt.show()

