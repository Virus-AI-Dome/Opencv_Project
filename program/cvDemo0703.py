"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【0703】直方图正规化
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def normalizedGrayhist(src):
    iMax, iMin = np.max(src), np.min(src)
    oMax, oMin = 255, 0
    a = float((oMax-oMin) / (iMax-iMin))
    b = oMin - a * iMin
    dst = a * src + b
    return dst.astype(np.uint8)  # 转为 CV-8U

if __name__ == '__main__':
    gray = cv.imread("../images/Fig0301.png", flags=0)  # 读取为灰度图像

    # 直方图规定化
    gray = cv.add(cv.multiply(gray, 0.6), 36)  # 调整灰度范围
    grayNorm1 = normalizedGrayhist(gray)  # 直方图规定化子函数
    grayNorm2 = cv.normalize(gray, None, 0, 255, cv.NORM_MINMAX)  # OpenCV 函数

    plt.figure(figsize=(9, 6))
    plt.subplot(231), plt.title("(1) Original"), plt.axis('off')
    plt.imshow(gray, cmap='gray')
    plt.subplot(232), plt.title("(2) Normalized"), plt.axis('off')
    plt.imshow(grayNorm1, cmap='gray')
    plt.subplot(233), plt.title("(3) cv.normalize"), plt.axis('off')
    plt.imshow(grayNorm2, cmap='gray')
    plt.subplot(234), plt.title("(4) Original histogram"), plt.axis('off')
    histCV = cv.calcHist([gray], [0], None, [256], [0, 255])  # 计算灰度直方图
    plt.bar(range(256), histCV[:, 0])  # 绘制灰度直方图
    plt.axis([0, 255, 0, np.max(histCV)])
    plt.subplot(235), plt.title("(5) Normalized histogram"), plt.axis('off')
    histCV1 = cv.calcHist([grayNorm1], [0], None, [256], [0, 255])
    plt.bar(range(256), histCV1[:, 0])
    plt.axis([0, 255, 0, np.max(histCV)])
    plt.subplot(236), plt.title("(6) cv.normalize histogram"), plt.axis('off')
    histCV2 = cv.calcHist([grayNorm2], [0], None, [256], [0, 255])
    plt.bar(range(256), histCV2[:, 0])
    plt.axis([0, 255, 0, np.max(histCV)])
    plt.tight_layout()
    plt.show()
