"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【1606】特征描述之 LBP 纹理特征符
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def getLBP1(gray):
    height, width = gray.shape
    dst = np.zeros((height, width), np.uint8)
    kernel = np.array([1, 2, 4, 128, 0, 8, 64, 32, 16]).reshape(3, 3)  # 从左上角开始顺时针旋转
    # kernel = np.array([64,128,1,32,0,2,16,8,4]).reshape(3,3)  # 从右上角开始顺时针旋转
    for h in range(1, height-1):
        for w in range(1, width-1):
            LBPMat = (gray[h-1:h+2, w-1:w+2] >= gray[h, w])  # 阈值比较
            dst[h, w] = np.sum(LBPMat * kernel)  # 二维矩阵相乘
    return dst

def getLBP2(gray):
    height, width = gray.shape
    dst = np.zeros((height, width), np.uint8)
    # kernelFlatten = np.array([1, 2, 4, 128, 0, 8, 64, 32, 16])  # 从左上角开始顺时针旋转
    kernelFlatten = np.array([64, 128, 1, 32, 0, 2, 16, 8, 4])  # 从右上角开始顺时针旋转
    for h in range(1, height-1):
        for w in range(1, width-1):
            LBPFlatten = (gray[h-1:h+2, w-1:w+2] >= gray[h, w]).flatten()  # 展平为一维向量, (9,)
            dst[h, w] = np.vdot(LBPFlatten, kernelFlatten)  # 一维向量的内积
    return dst

if __name__ == '__main__':
    img = cv.imread("../images/Fig1604.png", flags=1)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 灰度图像

    # LBP 基本算法：选取中心点周围的 8个像素点，阈值处理后标记为 8位二进制数
    # 1) 二重循环, 二维矩阵相乘
    timeBegin = cv.getTickCount()
    imgLBP1 = getLBP1(gray)  # # 从左上角开始顺时针旋转
    timeEnd = cv.getTickCount()
    time = (timeEnd - timeBegin)/cv.getTickFrequency()
    print("(1) 二重循环, 二维矩阵相乘: {} sec".format(round(time, 4)))

    # 2) 二重循环, 一维向量的内积
    timeBegin = cv.getTickCount()
    imgLBP2 = getLBP2(gray)  # 从右上角开始顺时针旋转
    timeEnd = cv.getTickCount()
    time = (timeEnd - timeBegin)/cv.getTickFrequency()
    print("(2) 二重循环, 一维向量的内积: {} sec".format(round(time, 4)))

    # 3) skimage 特征检测
    from skimage.feature import local_binary_pattern
    timeBegin = cv.getTickCount()
    imgLBP3 = local_binary_pattern(gray, 8, 1)
    timeEnd = cv.getTickCount()
    time = (timeEnd - timeBegin)/cv.getTickFrequency()
    print("(3) skimage.feature 封装: {} sec".format(round(time, 4)))

    plt.figure(figsize=(9, 3.3))
    plt.subplot(131), plt.title("(1) LBP(TopLeft)")
    plt.axis('off'), plt.imshow(imgLBP1, 'gray')
    plt.subplot(132), plt.title("(2) LBP(TopRight)")
    plt.axis('off'), plt.imshow(imgLBP2, 'gray')
    plt.subplot(133), plt.title("(3) LBP(SKimage)")
    plt.axis('off'), plt.imshow(imgLBP3, 'gray')
    plt.tight_layout()
    plt.show()
