"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【0903】图像阈值处理之 OTSU 算法
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv.imread("../images/Fig0901.png", flags=0)

    # OTSU 算法的实现
    histCV = cv.calcHist([img], [0], None, [256], [0, 256])  # 灰度直方图
    scale = range(256)  # 灰度级 [0,255]
    totalPixels = img.shape[0] * img.shape[1]  # 像素总数
    totalGray = np.dot(histCV[:, 0], scale)  # 内积, 总和灰度值
    mG = totalGray / totalPixels  # 平均灰度
    icv = np.zeros(256)
    numFt, sumFt = 0, 0
    for t in range(256):  # 遍历灰度值
        numFt += histCV[t, 0]  # F(t) 像素数量
        sumFt += histCV[t, 0] * t  # F(t) 灰度值总和
        pF = numFt / totalPixels  # F(t) 像素数占比
        mF = (sumFt / numFt) if numFt > 0 else 0  # F(t) 平均灰度
        numBt = totalPixels - numFt  # B(t) 像素数量
        sumBt = totalGray - sumFt  # B(t) 灰度值总和
        pB = numBt / totalPixels  # B(t) 像素数占比
        mB = (sumBt / numBt) if numBt > 0 else 0  # B(t) 平均灰度
        icv[t] = pF * (mF - mG) ** 2 + pB * (mB - mG) ** 2  # 灰度 t 的类间方差
    maxIcv = max(icv)  # ICV 的最大值
    maxIndex = np.argmax(icv)  # 最大值的索引，即为 OTSU阈值
    _, imgBin = cv.threshold(img, maxIndex, 255, cv.THRESH_BINARY)  # 以 maxIndex 作为最优阈值

    # cv.threshold 实现 OTSU 算法
    ret, imgOtsu = cv.threshold(img, 128, 255, cv.THRESH_OTSU)  # 阈值分割, OTSU
    print("maxIndex={}, retOtsu={}".format(maxIndex, round(ret)))

    plt.figure(figsize=(9, 3.5))
    plt.subplot(131), plt.axis('off'), plt.title("(1) Original"), plt.imshow(img, 'gray')
    plt.subplot(132), plt.title("(2) OTSU by ICV (T={})".format(maxIndex))
    plt.imshow(imgBin, 'gray'), plt.axis('off')
    plt.subplot(133), plt.title("(3) OTSU by OpenCV(T={})".format(round(ret)))
    plt.imshow(imgOtsu, 'gray'), plt.axis('off')
    plt.tight_layout()
    plt.show()
