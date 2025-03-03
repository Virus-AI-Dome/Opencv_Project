"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【0902】图像阈值处理之计算全局阈值
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv.imread("../images/Fig0302.png", flags=0)

    deltaT = 1  # 预定义值
    histCV = cv.calcHist([img], [0], None, [256], [0, 256])  # 灰度直方图
    grayScale = range(256)  # 灰度级 [0,255]
    totalPixels = img.shape[0] * img.shape[1]  # 像素总数
    totalGary = np.dot(histCV[:, 0], grayScale)  # 内积, 总和灰度值
    T = round(totalGary / totalPixels)  # 平均灰度作为阈值初值
    while True:  # 迭代计算分割阈值
        numC1 = np.sum(histCV[:T, 0])  # C1 像素数量
        sumC1 = np.sum(histCV[:T, 0] * range(T))  # C1 灰度值总和
        numC2 = totalPixels - numC1  # C2 像素数量
        sumC2 = totalGary - sumC1  # C2 灰度值总和
        T1 = round(sumC1 / numC1)  # C1 平均灰度
        T2 = round(sumC2 / numC2)  # C2 平均灰度
        Tnew = round((T1 + T2) / 2)  # 计算新的阈值
        print("T={}, m1={}, m2={}, Tnew={}".format(T, T1, T2, Tnew))
        if abs(T - Tnew) < deltaT:  # 等价于 T==Tnew
            break
        else:
            T = Tnew

    # 阈值处理
    ret, imgBin = cv.threshold(img, T, 255, cv.THRESH_BINARY)  # 阈值分割, thresh=T

    plt.figure(figsize=(9, 3.3))
    plt.subplot(131), plt.axis('off'), plt.title("(1) Original"), plt.imshow(img, 'gray')
    plt.subplot(132, yticks=[]), plt.title("(2) Gray Hist")  # 灰度直方图
    plt.bar(range(256), histCV[:, 0]), plt.axis([0, 255, 0, np.max(histCV)])
    plt.axvline(T, color='r', linestyle='--')  # 绘制固定阈值
    plt.text(T+5, 0.9*np.max(histCV), "T={}".format(T), fontsize=10)
    plt.subplot(133), plt.title("(3) Binary (T={})".format(T)), plt.axis('off')
    plt.imshow(imgBin, 'gray')
    plt.tight_layout()
    plt.show()
