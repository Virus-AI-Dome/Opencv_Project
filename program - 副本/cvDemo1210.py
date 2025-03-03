"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【1210】形态学算法之提取水平和垂直线
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv.imread("../images/Fig1001.png", flags=0)  # 读取为灰度图像
    _, imgBin = cv.threshold(img, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)  # 二值处理
    h, w = imgBin.shape[0], imgBin.shape[1]

    # 提取水平线
    hline = cv.getStructuringElement(cv.MORPH_RECT, ((w//16), 1), (-1, -1))  # 水平结构元
    imgOpenHline = cv.morphologyEx(imgBin, cv.MORPH_OPEN, hline)  # 开运算提取水平结构
    imgHline = cv.bitwise_not(imgOpenHline)  # 恢复白色背景

    # 提取垂直线
    vline = cv.getStructuringElement(cv.MORPH_RECT, (1, (h//16)), (-1, -1))  # 垂直结构元
    imgOpenVline = cv.morphologyEx(imgBin, cv.MORPH_OPEN, vline)  # 开运算提取垂直结构
    imgVline = cv.bitwise_not(imgOpenVline)

    # 删除水平线和垂直线
    lineRemoved = imgBin - imgOpenHline  # 删除水平线 (白底为 0)
    lineRemoved = lineRemoved - imgOpenVline  # 删除垂直线
    imgRebuild = cv.bitwise_not(lineRemoved)  # 恢复白色背景

    plt.figure(figsize=(9, 2.7))
    plt.subplot(141), plt.axis('off'), plt.title("(1) Original")
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.subplot(142), plt.title("(2) Horizontal line"), plt.axis('off')
    plt.imshow(imgHline, cmap='gray', vmin=0, vmax=255)
    plt.subplot(143), plt.title("(3) Vertical line"), plt.axis('off')
    plt.imshow(imgVline, cmap='gray', vmin=0, vmax=255)
    plt.subplot(144), plt.title("(4) H/V line removed"), plt.axis('off')
    plt.imshow(imgRebuild, cmap='gray', vmin=0, vmax=255)
    plt.tight_layout()
    plt.show()
