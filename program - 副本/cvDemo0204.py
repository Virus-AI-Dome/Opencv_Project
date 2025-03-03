"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【0204】图像通道的拆分与合并
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    filepath = "../images/Lena.tif"  # 读取文件的路径
    img = cv.imread(filepath,flags=1)  # flags=1 读取彩色图像(BGR)
    cv.imshow('Lena', img)
    cv.waitKey(0)
    # (1) cv.split 实现图像通道的拆分
    bImg, gImg, rImg = cv.split(img)  # 拆分为 BGR 独立通道
    # (2) cv.merge 实现图像通道的合并
    imgMerge = cv.merge([bImg, gImg, rImg])
    # (3) Numpy 拼接实现图像通道的合并
    imgStack = np.stack((bImg, gImg, rImg), axis=2)
    # (4) Numpy 切片提取颜色分量
    # 提取 B 通道
    imgB = img.copy()  # BGR
    imgB[:, :, 1] = 0  # G=0
    imgB[:, :, 2] = 0  # R=0
    # 提取 G 通道
    imgG = img.copy()  # BGR
    imgG[:, :, 0] = 0  # B=0
    imgG[:, :, 2] = 0  # R=0
    # 提取 R 通道
    imgR = img.copy()  # BGR
    imgR[:, :, 0] = 0  # B=0
    imgR[:, :, 1] = 0  # G=0
    # 消除 B 通道（保留 G/R 通道）
    imgGR = img.copy()  # BGR
    imgGR[:, :, 0] = 0  # B=0

    plt.figure(figsize=(8, 7))
    plt.subplot(221), plt.title("(1) B channel"), plt.axis('off')
    plt.imshow(cv.cvtColor(imgB, cv.COLOR_BGR2RGB))  # 显示 B 通道
    plt.subplot(222), plt.title("(2) G channel"), plt.axis('off')
    plt.imshow(cv.cvtColor(imgG, cv.COLOR_BGR2RGB))  # 显示 G 通道
    plt.subplot(223), plt.title("(3) R channel"), plt.axis('off')
    plt.imshow(cv.cvtColor(imgR, cv.COLOR_BGR2RGB))  # 显示 R 通道
    plt.subplot(224), plt.title("(4) GR channel"), plt.axis('off')
    plt.imshow(cv.cvtColor(imgGR, cv.COLOR_BGR2RGB))  # 显示 G/R 通道
    plt.tight_layout()
    plt.show()