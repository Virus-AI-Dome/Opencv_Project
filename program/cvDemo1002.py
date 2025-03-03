"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【1002】空间滤波之盒式低通滤波器
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv.imread("../images/Fig1001.png", flags=0)  # 读取灰度图像

    # (1) 盒式滤波器的 3 种实现方法
    ksize = (5, 5)
    kernel = np.ones(ksize, np.float32) / (ksize[0]*ksize[1])  # 生成归一化核
    conv1 = cv.filter2D(img, -1, kernel)  # cv.filter2D 卷积方法
    conv2 = cv.blur(img, ksize)  # cv.blur 函数
    conv3 = cv.boxFilter(img, -1, ksize)  # cv.boxFilter 函数
    print("Compare conv1 & conv2：", (conv1==conv2).all())
    print("Compare conv1 & conv3：", (conv1==conv3).all())

    # (2) 滤波器尺寸的影响
    imgConv1 = cv.blur(img, (5, 5))  # ksize=(5,5)
    imgConv2 = cv.blur(img, (11, 11))  # ksize=(11,11)

    plt.figure(figsize=(9, 3.2))
    plt.subplot(131), plt.axis('off'), plt.title("(1) Original")
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.subplot(132), plt.axis('off'), plt.title("(2) cv.boxFilter (5,5)")
    plt.imshow(imgConv1, cmap='gray', vmin=0, vmax=255)
    plt.subplot(133), plt.axis('off'), plt.title("(3) cv.boxFilter (11,11)")
    plt.imshow(imgConv2, cmap='gray', vmin=0, vmax=255)
    plt.tight_layout()
    plt.show()
