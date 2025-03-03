"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【0804】彩色图像的直方图匹配
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    imgSrc = cv.imread("../images/Fig0801.png", flags=1)  # 读取待匹配的图像
    imgRef = cv.imread("../images/Fig0301.png", flags=1)  # 读取模板图像

    imgDst = np.zeros_like(imgSrc)
    for i in range(imgSrc.shape[2]):
        # 计算累计直方图
        histSrc = cv.calcHist([imgSrc], [i], None, [256], [0, 255])  # (256,1)
        histRef = cv.calcHist([imgRef], [i], None, [256], [0, 255])  # (256,1)
        cdfSrc = np.cumsum(histSrc)  # 原始图像累积分布函数 CDF
        cdfRef = np.cumsum(histRef)  # 匹配模板累积分布函数 CDF
        cdfSrc = cdfSrc / cdfSrc[-1]  # 归一化: 0~1
        cdfRef = cdfRef / cdfRef[-1]
        for j in range(256):
            tmp = abs(cdfSrc[j] - cdfRef)
            tmp = tmp.tolist()
            index = tmp.index(min(tmp))
            imgDst[:,:,i][imgSrc[:,:,i]==j] = index

    color = ['b', 'g', 'r']
    fig = plt.figure(figsize=(9, 6))
    plt.subplot(231), plt.title("(1) Original"), plt.axis('off')
    plt.imshow(cv.cvtColor(imgSrc, cv.COLOR_BGR2RGB))  # 待匹配彩色图像
    plt.subplot(232), plt.title("(2) Matching template"), plt.axis('off')
    plt.imshow(cv.cvtColor(imgRef, cv.COLOR_BGR2RGB))  # 彩色模板图像
    plt.subplot(233), plt.title("(3) Matched result"), plt.axis('off')
    plt.imshow(cv.cvtColor(imgDst, cv.COLOR_BGR2RGB))  # 匹配结果图像
    plt.subplot(234, xticks=[], yticks=[])
    plt.title("(4) Original hist")
    for ch, col in enumerate(color):
        histCh = cv.calcHist([imgSrc], [ch], None, [256], [0, 255])
        histCh = histCh/np.max(histCh) + ch
        plt.plot(histCh, color=col)
        plt.xlim([0, 256])
    plt.subplot(235, xticks=[], yticks=[])
    plt.title("(5) Template hist")
    for ch, col in enumerate(color):
        histCh = cv.calcHist([imgRef], [ch], None, [256], [0, 255])
        histCh = histCh/np.max(histCh) + ch
        plt.plot(histCh, color=col)
        plt.xlim([0, 256])
    plt.subplot(236, xticks=[], yticks=[])
    plt.title("(6) Matched hist")
    for ch, col in enumerate(color):
        histCh = cv.calcHist([imgDst], [ch], None, [256], [0, 255])
        histCh = histCh/np.max(histCh) + ch
        plt.plot(histCh, color=col)
        plt.xlim([0, 256])
    plt.tight_layout()
    plt.show()
