"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【1214】泛洪算法实现孔洞填充
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv.imread("../images/Fig1205.png", flags=0)  # 灰度图像

    _, imgBinInv = cv.threshold(img, 205, 255, cv.THRESH_BINARY)  # 二值处理 (白色背景)
    imgBin = cv.bitwise_not(imgBinInv)  # 二值图像的补集 (黑色背景)，填充基准

    h, w = imgBin.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)  # 掩模比图像宽 2 个像素、高 2 个像素

    imgFloodfill = imgBin.copy()  # 输入孔洞图像，返回填充孔洞
    cv.floodFill(imgFloodfill, mask, (0,0), newVal=225)  # 从背景像素原点 (0,0) 开始
    imgRebuild = cv.bitwise_and(imgBinInv, imgFloodfill)  # 重建孔洞填充结果图像

    plt.figure(figsize=(9, 3))
    plt.subplot(131), plt.axis('off'), plt.title("(1) Binary invert")
    plt.imshow(imgBin, cmap='gray', vmin=0, vmax=255)
    plt.subplot(132), plt.title("(2) Filled holes"), plt.axis('off')
    plt.imshow(imgFloodfill, cmap='gray', vmin=0, vmax=255)
    plt.subplot(133), plt.title("(3) Rebuild image"), plt.axis('off')
    plt.imshow(imgRebuild, cmap='gray', vmin=0, vmax=255)
    plt.tight_layout()
    plt.show()
