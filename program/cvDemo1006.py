"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【1006】空间滤波之自适应局部降噪滤波器
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv.imread("../images/Fig1003.png", flags=0)  # 读取灰度图像
    hImg, wImg = img.shape[:2]
    print(hImg, wImg)

    # 边界填充
    m, n = 5, 5  # 滤波器尺寸，m*n矩形邻域
    hPad, wPad = int((m-1)/2), int((n-1)/2)
    imgPad = cv.copyMakeBorder(img, hPad, hPad, wPad, wPad, cv.BORDER_REFLECT)

    # 估计原始图像的噪声方差 varImg
    mean, stddev = cv.meanStdDev(img)  # 图像均值，方差
    varImg = stddev ** 2

    # 自适应局部降噪
    epsilon = 1e-8
    imgAdaptLocal = np.zeros(img.shape)
    for h in range(hImg):
        for w in range(wImg):
            neighborhood = imgPad[h:h+m, w:w+n]  # 邻域 Sxy，m*n
            meanSxy, stddevSxy = cv.meanStdDev(neighborhood)  # 邻域局部均值
            varSxy = stddevSxy**2  # 邻域局部方差
            ratioVar = min(varImg / (varSxy + epsilon), 1.0)  # 加性噪声 varImg<varSxy
            imgAdaptLocal[h,w] = img[h,w] - ratioVar * (img[h,w] - meanSxy)

    # 均值滤波器，用于比较
    imgAriMean = cv.boxFilter(img, -1, (m, n))

    plt.figure(figsize=(9, 3.5))
    plt.subplot(131), plt.axis('off'), plt.title("(1) Original")
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.subplot(132), plt.axis('off'), plt.title("(2) Box filter")
    plt.imshow(imgAriMean, cmap='gray', vmin=0, vmax=255)
    plt.subplot(133), plt.axis('off'), plt.title("(3) Adaptive local filter")
    plt.imshow(imgAdaptLocal, cmap='gray', vmin=0, vmax=255)
    plt.tight_layout()
    plt.show()
