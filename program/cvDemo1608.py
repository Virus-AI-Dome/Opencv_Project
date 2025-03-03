"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""


# 【1608】特征描述之 LBP 统计直方图
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def basicLBP(gray):
    height, width = gray.shape
    dst = np.zeros((height, width), np.uint8)
    kernelFlatten = np.array([1, 2, 4, 128, 0, 8, 64, 32, 16])  # 从左上角开始顺时针旋转
    for h in range(1, height-1):
        for w in range(1, width-1):
            LBPFlatten = (gray[h-1:h+2, w-1:w+2] >= gray[h, w]).flatten()  # 展平为一维向量, (9,)
            dst[h, w] = np.vdot(LBPFlatten, kernelFlatten)  # 一维向量的内积
    return dst

def calLBPHistogram(imgLBP, nCellX, nCellY):  # 计算 LBP 直方图
    height, width = gray.shape
    # nCellX, nCellY = 4, 4  # 将图像划分为 nCellX*nCellY 个子区域
    hCell, wCell = height // nCellY, width // nCellX  # 子区域的高度与宽度 (150,120)
    LBPHistogram = np.zeros((nCellX * nCellY, 256), np.int16)
    for j in range(nCellY):
        for i in range(nCellX):
            cell = imgLBP[j*hCell : (j+1)*hCell, i*wCell : (i+1)*wCell].copy()  # 子区域 cell LBP
            print("{}, Cell({}{}): [{}:{}, {}:{}]".format
                  (j*nCellX+i+1, j+1, i+1, j*hCell, (j+1)*hCell, i*wCell, (i+1)*wCell))
            histCell = cv.calcHist([cell], [0], None, [256], [0, 256])  # 子区域 LBP 直方图
            LBPHistogram[(i+1)*(j+1)-1, :] = histCell.flatten()
    print(LBPHistogram.shape)
    return LBPHistogram

if __name__ == '__main__':
    img = cv.imread("../images/Fig1605.png", flags=1)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 灰度图像
    height, width = gray.shape

    nCellX, nCellY = 3, 3  # 将图像划分为 nCellX*nCellY 个子区域
    hCell, wCell = height // nCellY, width // nCellX  # 子区域的高度与宽度 (150,120)
    print("img: h={},w={}, cell: h={},w={}".format(height, width, hCell, wCell))
    basicLBP = basicLBP(gray)  # 计算 basicLBP 描述符
    # LBPHistogram = calLBPHistogram(basicLBP, nCellX, nCellY)  # 计算 LBP 直方图 (16, 256)

    fig1 = plt.figure(figsize=(9, 7))
    fig1.suptitle("basic LBP")
    fig2 = plt.figure(figsize=(9, 7))
    fig2.suptitle("LBP histogram")
    for j in range(nCellY):
        for i in range(nCellX):
            cell = basicLBP[j*hCell : (j+1)*hCell, i*wCell : (i+1)*wCell].copy()  # 子区域 cell LBP
            histCV = cv.calcHist([cell], [0], None, [256], [0,256])  # 子区域 cell LBP 直方图
            ax1 = fig1.add_subplot(nCellY, nCellX, j*nCellX+i+1)
            ax1.set_xticks([]), ax1.set_yticks([])
            ax1.imshow(cell, 'gray')  # 绘制子区域 LBP
            ax2 = fig2.add_subplot(nCellY, nCellX, j*nCellX+i+1)
            ax2.set_xticks([]), ax2.set_yticks([])
            ax2.bar(range(256), histCV[:, 0])  # 绘制子区域LBP 直方图
            print("{}, Cell({}{}): [{}:{}, {}:{}]".format
                  (j*nCellX+i+1, j+1, i+1, j*hCell, (j+1)*hCell, i*wCell, (i+1)*wCell))
    plt.tight_layout()
    plt.show()

