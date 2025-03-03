"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

#【1503】超像素区域分割
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    # 注意：本例程需要 opencv-contrib-python 包的支持
    img = cv.imread("../images/Fig0301.png", flags=1)  # 彩色图像(BGR)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    imgHSV = cv.cvtColor(img, cv.COLOR_BGR2HSV_FULL)  # BGR-HSV 转换

    # (1) SLIC 算法
    slic = cv.ximgproc.createSuperpixelSLIC(imgHSV, region_size=20, ruler=10.0)  # 初始化 SLIC
    slic.iterate(5)  # 迭代次数
    slic.enforceLabelConnectivity(50)  # 最小尺寸
    maskSlic = slic.getLabelContourMask()  # 超像素边缘 Mask==1
    imgSlic = cv.bitwise_and(img, img, mask=cv.bitwise_not(maskSlic))  # 绘制超像素边界
    numberSlic = slic.getNumberOfSuperpixels()  # 超像素数目
    imgSlicW = cv.add(gray, maskSlic)
    print("number of SLICO", numberSlic)

    # (2) SEEDS 算法，注意图片长宽的顺序为 w, h, c
    w, h, c = imgHSV.shape[1], imgHSV.shape[0], imgHSV.shape[2]
    seeds = cv.ximgproc.createSuperpixelSEEDS(w, h, c, 2000, 15, 3, 5)
    seeds.iterate(imgHSV, 5)  # 输入图像大小必须与初始化形状相同
    maskSeeds = seeds.getLabelContourMask()  # 超像素边缘 Mask==1
    labelSeeds = seeds.getLabels()  # 获取超像素标签
    numberSeeds = seeds.getNumberOfSuperpixels()  # 获取超像素数目
    imgSeeds = cv.bitwise_and(img, img, mask=cv.bitwise_not(maskSeeds))
    imgSeedsW = cv.add(gray, maskSeeds)
    print("number of SEEDS", numberSeeds)

    # (3) LSC 算法 (Linear Spectral Clustering)
    lsc = cv.ximgproc.createSuperpixelLSC(img, region_size=20)
    lsc.iterate(5)  # 迭代次数
    maskLsc = lsc.getLabelContourMask()  # 超像素边缘 Mask==0
    labelLsc = lsc.getLabels()  # 超像素标签
    numberLsc = lsc.getNumberOfSuperpixels()  # 超像素数目
    imgLsc = cv.bitwise_and(img, img, mask=cv.bitwise_not(maskLsc))
    imgLscW = cv.add(gray, maskLsc)
    print("number of LSC", numberLsc)

    plt.figure(figsize=(9, 3.5))
    plt.subplot(131), plt.axis('off'), plt.title("(1) SLIC image")
    plt.imshow(imgSlicW, cmap='gray')
    plt.subplot(132), plt.axis('off'), plt.title("(2) SEEDS image")
    plt.imshow(imgSeedsW, cmap='gray')
    plt.subplot(133), plt.axis('off'), plt.title("(3) LSC image")
    plt.imshow(imgLscW, cmap='gray')
    plt.tight_layout()
    plt.show()

