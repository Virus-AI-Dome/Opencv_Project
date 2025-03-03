"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

#【1405】边缘连接的局部处理简化方法
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv.imread("../images/Fig1201.png", flags=0)  # flags=0 读取为灰度图像
    hImg, wImg = img.shape

    # (1) Sobel 计算梯度
    gx = cv.Sobel(img, cv.CV_32F, 1, 0, ksize=3)  # SobelX 水平梯度
    gy = cv.Sobel(img, cv.CV_32F, 0, 1, ksize=3)  # SobelY 垂直梯度
    mag, angle = cv.cartToPolar(gx, gy, angleInDegrees=1)  # 梯度幅值和角度
    angle = np.abs(angle-180)  # 角度转换 (0,360)->(0,180)
    print(mag.max(), mag.min(), angle.max(), angle.min())

    # (2) 边缘像素的相似性判断
    TM = 0.2 * mag.max()  # TM 设为最大梯度的 20%
    A, Ta = 90, 44  # A=90 水平扫描, Ta = 30
    edgeX = np.zeros((hImg, wImg), np.uint8)  # 水平边缘
    X, Y = np.where((mag>TM) & (angle>=A-Ta) & (angle<=A+Ta))  # 幅值和角度条件
    edgeX[X, Y] = 255  # 水平边缘二值化处理
    edgeY = np.zeros((hImg, wImg), np.uint8)  # 垂直边缘
    X, Y = np.where((mag>TM) & ((angle<=Ta) | (angle>=180-Ta)))  # 幅值和角度条件
    edgeY[X, Y] = 255  # 垂直边缘二值化处理

    # (3) 合成水平边缘与垂直边缘
    edgeConnect = cv.bitwise_or(edgeX, edgeY)

    plt.figure(figsize=(9, 3.3))
    plt.subplot(131), plt.title("(1) Original")
    plt.axis('off'), plt.imshow(img, cmap='gray')
    plt.subplot(132), plt.title("(2) Gradient magnitude")
    plt.axis('off'), plt.imshow(np.uint8(mag), cmap='gray')
    plt.subplot(133), plt.title("(3) Edge connect")
    plt.axis('off'), plt.imshow(edgeConnect, cmap='gray')
    plt.tight_layout()
    plt.show()