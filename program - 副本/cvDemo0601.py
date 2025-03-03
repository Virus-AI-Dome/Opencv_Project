"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【0601】图像的平移
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv.imread("../images/Lena.tif")  # 读取彩色图像(BGR)
    height, width = img.shape[:2]

    dx, dy = 100, 50  # dx 向右平移, dy 向下平移
    MAT = np.float32([[1, 0, dx], [0, 1, dy]])  # 构造平移变换矩阵
    imgTrans1 = cv.warpAffine(img, MAT, (width, height))
    imgTrans2 = cv.warpAffine(img, MAT, (601, 401), borderValue=(255,255,255))

    plt.figure(figsize=(9, 3.3))
    plt.subplot(131), plt.title("(1) Original"), plt.axis('off')
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.subplot(132), plt.title("(2) Translation 1")
    plt.imshow(cv.cvtColor(imgTrans1, cv.COLOR_BGR2RGB))
    plt.subplot(133), plt.title("(3) Translation 2")
    plt.imshow(cv.cvtColor(imgTrans2, cv.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()



