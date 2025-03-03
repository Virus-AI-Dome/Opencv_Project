"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【1005】空间滤波之统计排序滤波器
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv.imread("../images/Fig1002.png", flags=0)  # 读取灰度图像
    hImg, wImg = img.shape[:2]

    # 边界填充
    m, n = 3, 3  # 统计滤波器尺寸
    hPad, wPad = int((m-1)/2), int((n-1)/2)
    imgPad = cv.copyMakeBorder(img, hPad, hPad, wPad, wPad, cv.BORDER_REFLECT)

    imgMedianF = np.zeros(img.shape)  # 中值滤波器
    imgMaximumF = np.zeros(img.shape)  # 最大值滤波器
    imgMinimumF = np.zeros(img.shape)  # 最小值滤波器
    imgMiddleF = np.zeros(img.shape)  # 中点滤波器
    imgAlphaF = np.zeros(img.shape)  # 修正阿尔法均值滤波器
    for h in range(hImg):
        for w in range(wImg):
            # 当前像素的邻域
            neighborhood = imgPad[h:h+m, w:w+n]
            padMax = np.max(neighborhood)  # 邻域最大值
            padMin = np.min(neighborhood)  # 邻域最小值
            # (1) 中值滤波器 (median filter)
            imgMedianF[h,w] = np.median(neighborhood)
            # (2) 最大值滤波器 (maximum filter)
            imgMaximumF[h,w] = padMax
            # (3) 最小值滤波器 (minimum filter)
            imgMinimumF[h,w] = padMin
            # (4) 中点滤波器 (middle filter)
            imgMiddleF[h,w] = int(padMax/2 + padMin/2)
            # 注意不能写成 int((padMax+padMin)/2)，以免数据溢出
            # (5) 修正阿尔法均值滤波器 (Modified alpha-mean filter)
            d = 2  # 修正值
            neighborSort = np.sort(neighborhood.flatten())  # 对邻域像素按灰度值排序
            sumAlpha = np.sum(neighborSort[d:m*n-d-1])  # 删除 d 个最大灰度值, d 个最小灰度值
            imgAlphaF[h,w] = sumAlpha / (m*n-2*d)  # 对剩余像素进行算术平均

    plt.figure(figsize=(9, 6.5))
    plt.subplot(231), plt.axis('off'), plt.title("(1) Original")
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.subplot(232), plt.axis('off'), plt.title("(2) Median filter")
    plt.imshow(imgMedianF, cmap='gray', vmin=0, vmax=255)
    plt.subplot(233), plt.axis('off'), plt.title("(3) Maximum filter")
    plt.imshow(imgMaximumF, cmap='gray', vmin=0, vmax=255)
    plt.subplot(234), plt.axis('off'), plt.title("(4) Minimum filter")
    plt.imshow(imgMinimumF, cmap='gray', vmin=0, vmax=255)
    plt.subplot(235), plt.axis('off'), plt.title("(5) Middle filter")
    plt.imshow(imgMiddleF, cmap='gray', vmin=0, vmax=255)
    plt.subplot(236), plt.axis('off'), plt.title("(6) Modified alpha-mean")
    plt.imshow(imgAlphaF, cmap='gray', vmin=0, vmax=255)
    plt.tight_layout()
    plt.show()
