"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【1001】图像的卷积运算与相关运算
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv.imread("../images/Fig1001.png", flags=0)  # 读取灰度图像

    # (1) 不对称卷积核
    kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # 不对称核
    imgCorr = cv.filter2D(img, -1, kernel)  # 相关运算
    kernFlip = cv.flip(kernel, -1)  # 翻转卷积核
    imgConv = cv.filter2D(img, -1, kernFlip)  # 卷积运算
    print("(1) Asymmetric convolution kernel")
    print("\tCompare imgCorr & imgConv：", (imgCorr==imgConv).all())
    # (2) 对称卷积核
    kernSymm = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])  # 对称核
    imgCorrSym = cv.filter2D(img, -1, kernSymm)
    kernFlip = cv.flip(kernSymm, -1)  # 卷积核旋转180度
    imgConvSym = cv.filter2D(img, -1, kernFlip)
    print("(2) Symmetric convolution kernel")
    print("\tCompare imgCorr & imgConv：", (imgCorrSym==imgConvSym).all())

    # (3) 可分离卷积核: kernXY = kernX * kernY
    kernX = np.array([[-1, 2, -1]], np.float32)  # 水平卷积核 (1,3)
    kernY = np.transpose(kernX)  # 垂直卷积核(3,1)
    kernXY = kernX * kernY  # 二维卷积核 (3,3)
    kFlip = cv.flip(kernXY, -1)  # 水平和垂直翻转卷积核
    imgConvXY = cv.filter2D(img, -1, kernXY)  # 直接使用二维卷积核
    imgConvSep = cv.sepFilter2D(img, -1, kernX, kernY)  # 分离核依次卷积
    print("(3) Separable convolution kernel")
    print("\tCompare imgConvXY & imgConvSep：", (imgConvXY==imgConvSep).all())
    print("kernX:{}, kernY:{}, kernXY:{}".format(kernX.shape, kernY.shape, kernXY.shape))

    plt.figure(figsize=(9, 3.2))
    plt.subplot(131), plt.axis('off'), plt.title("(1) Original")
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.subplot(132), plt.axis('off'), plt.title("(2) Correlation")
    plt.imshow(imgCorr, cmap='gray', vmin=0, vmax=255)
    plt.subplot(133), plt.axis('off'), plt.title("(3) Convolution")
    plt.imshow(imgConv, cmap='gray', vmin=0, vmax=255)
    plt.tight_layout()
    plt.show()


