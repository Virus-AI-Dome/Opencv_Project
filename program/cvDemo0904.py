"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【0904】阈值处理之多阈值 OTSU 方法
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def doubleThreshold(img):
    histCV = cv.calcHist([img], [0], None, [256], [0, 256])  # 灰度直方图
    grayScale = np.arange(0, 256, 1)  # 灰度级 [0,255]
    totalPixels = img.shape[0] * img.shape[1]  # 像素总数
    totalGray = np.dot(histCV[:, 0], grayScale)  # 内积, 总和灰度值
    mG = totalGray / totalPixels  # 平均灰度，meanGray
    varG = sum(((i-mG)**2 * histCV[i, 0] / totalPixels) for i in range(256))

    T1, T2, varMax = 1, 2, 0.0
    for k1 in range(1, 254):  # k1: [1,253], 1<=k1<k2<=254
        n1 = sum(histCV[:k1, 0])  # C1 像素数量
        s1 = sum((i * histCV[i, 0]) for i in range(k1))
        P1 = n1 / totalPixels  # C1 像素数占比
        m1 = (s1 / n1) if n1 > 0 else 0  # C1 平均灰度
        for k2 in range(k1 + 1, 256):  # k2: [2,254], k2>k1
            n3 = sum(histCV[k2 + 1:, 0])  # C3 像素数量
            s3 = sum((i * histCV[i, 0]) for i in range(k2 + 1, 256))
            P3 = n3 / totalPixels  # C3 像素数占比
            m3 = (s3 / n3) if n3 > 0 else 0  # C3 平均灰度
            P2 = 1.0 - P1 - P3  # C2 像素数占比
            m2 = (mG - P1*m1 - P3*m3) / P2 if P2 > 1e-6 else 0  # C2 平均灰度
            var = P1 * (m1-mG)** 2 + P2 * (m2-mG)**2 + P3 * (m3-mG)**2
            if var > varMax:
                T1, T2, varMax = k1, k2, var

    epsT = varMax / varG  # 可分离测度
    print(totalPixels, mG, varG, varMax, epsT, T1, T2)
    return T1, T2, epsT

if __name__ == '__main__':
    img = cv.imread("../images/Fig0901.png", flags=0)

    # OTSU 方法计算多阈值处理的阈值
    T1, T2, epsT = doubleThreshold(img)  # 多阈值处理子程序
    print("T1={}, T2={}, esp={:.4f}".format(T1, T2, epsT))
    # 基于 OTSU 最优阈值进行多阈值处理
    imgMClass = img.copy()
    imgMClass[img < T1] = 0
    imgMClass[img > T2] = 255

    # 不同阈值处理方法的对比
    ret, imgOtsu = cv.threshold(img, 127, 255, cv.THRESH_OTSU)  # OTSU 阈值分割
    _, binary1 = cv.threshold(img, T1, 255, cv.THRESH_BINARY)  # 小于阈值置 0，大于阈值不变
    _, binary2 = cv.threshold(img, T2, 255, cv.THRESH_BINARY)

    plt.figure(figsize=(9, 6))
    plt.subplot(231), plt.axis('off'), plt.title("(1) Original"), plt.imshow(img, 'gray')
    histCV = cv.calcHist([img], [0], None, [256], [0, 256])  # 灰度直方图
    plt.subplot(232, yticks=[]), plt.axis([0, 255, 0, np.max(histCV)])
    plt.bar(range(256), histCV[:, 0]), plt.title("(2) Gray Hist")
    plt.subplot(233), plt.title("(3) Double Thresh({},{})".format(T1, T2))
    plt.axis('off'), plt.imshow(imgMClass, 'gray')
    plt.subplot(234), plt.title("(4) Threshold(T={})".format(T1))
    plt.axis('off'), plt.imshow(binary1, 'gray')
    plt.subplot(235), plt.title("(5) OTSU Thresh(T={})".format(round(ret)))
    plt.axis('off'), plt.imshow(imgOtsu, 'gray')
    plt.subplot(236), plt.title("(6) Threshold(T={})".format(T2))
    plt.axis('off'), plt.imshow(binary2, 'gray')
    plt.tight_layout()
    plt.show()
