"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【0805】基于局部直方图统计量的图像增强
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    gray = cv.imread("../images/Fig0803.png", flags=0)  # 读取灰度图像
    height, width = gray.shape
    imgROI = gray[20:40, 20:40]
    maxGray, maxROI = gray.max(), imgROI.max()
    const = maxGray / maxROI

    m = 5
    half = 3
    k0, k1, k2, k3 = 0.0, 0.2, 0.0, 0.2
    meanG = np.mean(gray)  # 全局均值
    sigmaG = np.std(gray)  # 全局标准差
    minMeanG, maxMeanG = int(k0*meanG), int(k1*meanG)
    minSigmaG, maxSigmaG = int(k2*sigmaG), int(k3*sigmaG)
    print(minMeanG, maxMeanG, minSigmaG, maxSigmaG)

    imgEnhance = gray.copy()
    for h in range(half, height-half-1):
        for w in range(half, width-half-1):
            sxy = gray[h-half:h+half+1, w-half:w+half+1]
            meanSxy = int(np.mean(sxy))
            sigmaSxy = int(np.std(sxy))
            if minMeanG <= meanSxy <= maxMeanG and minSigmaG <= sigmaSxy <= maxSigmaG:
                imgEnhance[h + half, w + half] = int(const * sxy[half, half])
            else:
                imgEnhance[h + half, w + half] = sxy[half, half]

    plt.figure(figsize=(9, 3.5))
    plt.subplot(131), plt.title("(1) Original"), plt.axis('off')
    plt.imshow(gray, cmap='gray', vmin=0, vmax=255)
    plt.subplot(132), plt.title("(2) Global equalize"), plt.axis('off')
    equalize = cv.equalizeHist(gray)  # 直方图均衡
    plt.imshow(equalize, cmap='gray', vmin=0, vmax=255)
    plt.subplot(133), plt.title("(3) LocalHist enhancement "), plt.axis('off')
    plt.imshow(imgEnhance, cmap='gray', vmin=0, vmax=255)
    plt.tight_layout()
    plt.show()
