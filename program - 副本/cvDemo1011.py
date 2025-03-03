"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【1011】空间滤波之 Sobel 算子
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv.imread("../images/Fig1001.png", flags=0)
    print(img.shape[:2])

    # (1) 使用函数 filter2D 实现
    kernSobelX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernSobelY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    SobelX = cv.filter2D(img, -1, kernSobelX)
    SobelY = cv.filter2D(img, -1, kernSobelY)

    # (2) 使用 cv.Sobel 实现
    imgSobelX = cv.Sobel(img, cv.CV_16S, 1, 0)  # X 轴方向
    imgSobelY = cv.Sobel(img, cv.CV_16S, 0, 1)  # Y 轴方向
    absSobelX = cv.convertScaleAbs(imgSobelX)  # 转回 uint8
    absSobelY = cv.convertScaleAbs(imgSobelY)  # 转回 uint8
    SobelXY = cv.add(absSobelX, absSobelY)  # 用绝对值近似平方根

    plt.figure(figsize=(9, 6))
    plt.subplot(231), plt.axis('off'), plt.title("(1) Original")
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.subplot(232), plt.axis('off'), plt.title("(2) SobelX(float)")
    plt.imshow(imgSobelX, cmap='gray', vmin=0, vmax=255)
    plt.subplot(233), plt.axis('off'), plt.title("(3) SobelY(float)")
    plt.imshow(imgSobelY, cmap='gray', vmin=0, vmax=255)
    plt.subplot(234), plt.axis('off'), plt.title("(4) SobelXY(abs)")
    plt.imshow(SobelXY, cmap='gray')
    plt.subplot(235), plt.axis('off'), plt.title("(5) SobelX(abs)")
    plt.imshow(absSobelX, cmap='gray', vmin=0, vmax=255)
    plt.subplot(236), plt.axis('off'), plt.title("(6) SobelY(abs)")
    plt.imshow(absSobelY, cmap='gray', vmin=0, vmax=255)
    plt.tight_layout()
    plt.show()
