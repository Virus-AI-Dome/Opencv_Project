"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【0906】阈值处理之移动平均方法
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def movingThreshold(img, n, b):
    imgFlip = img.copy()
    imgFlip[1:-1:2, :] = np.fliplr(img[1:-1:2, :])  # 向量翻转
    f = imgFlip.flatten()  # 展平为一维
    ret = np.cumsum(f)
    ret[n:] = ret[n:] - ret[:-n]
    m = ret / n  # 移动平均值
    g = np.array(f >= b * m).astype(int)  # 阈值判断，g=1 if f>=b*m
    g = g.reshape(img.shape)  # 恢复为二维
    g[1:-1:2, :] = np.fliplr(g[1:-1:2, :])  # 交替翻转
    return g * 255

if __name__ == '__main__':
    img = cv.imread("../images/Fig0902.png", flags=0)

    # OTSU 阈值分割
    ret1, imgOtsu1 = cv.threshold(img, 127, 255, cv.THRESH_OTSU)
    # 移动平均阈值处理：n=8, b=0.8
    imgMoveThres1 = movingThreshold(img, 8, 0.8)

    plt.figure(figsize=(9, 3))
    plt.subplot(131), plt.title("(1) Original")
    plt.axis('off'), plt.imshow(img, 'gray')
    plt.subplot(132), plt.title("(2) OTSU (T={})".format(ret1))
    plt.axis('off'), plt.imshow(imgOtsu1, 'gray')
    plt.subplot(133), plt.title("(3) MA threshold")
    plt.axis('off'), plt.imshow(imgMoveThres1, 'gray', vmin=0, vmax=255)
    plt.tight_layout()
    plt.show()

