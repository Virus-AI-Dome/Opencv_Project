"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【` `30501】图像的加法运算
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':

    img1 = cv.imread('../images/Lena.tif')
    img2 = cv.imread('../images/Fig0301.png')

    # h,w 行列
    h, w = img1.shape[:2]
    # dsize w,h 高宽
    img3 = cv.resize(img2, (w, h))
    gray = cv.cvtColor(img3, cv.COLOR_BGR2GRAY)
    print("img1:{}, img2:{}, img3:{},gray:{}", img1.shape, img2.shape, img3.shape, gray.shape)

    value = 100

    imgAddV = cv.add(img1, value)
    imgAddG = cv.add(gray, value)
    # (2) 彩色图像和标量相加
    scalar = np.ones((1, 3)) * value
    imgAddS = cv.add(img1, scalar)
    # (3) Numpy 取模加法
    imgAddNP = img1 + img3
    # (4) Opencv 饱和加法
    imgAddCV = cv.add(img1, img3)

    plt.figure(figsize=(9, 6))
    plt.subplot(231), plt.title("(1) img1"), plt.axis('off')
    plt.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
    plt.subplot(232), plt.title("(2) add(img, value)"), plt.axis('off')
    plt.imshow(cv.cvtColor(imgAddV, cv.COLOR_BGR2RGB))
    plt.subplot(233), plt.title("(3) add(img, scalar)"), plt.axis('off')
    plt.imshow(cv.cvtColor(imgAddS, cv.COLOR_BGR2RGB))
    plt.subplot(234), plt.title("(4) img3"), plt.axis('off')
    plt.imshow(cv.cvtColor(img2, cv.COLOR_BGR2RGB))
    plt.subplot(235), plt.title("(5) img1 + img3"), plt.axis('off')
    plt.imshow(cv.cvtColor(imgAddNP, cv.COLOR_BGR2RGB))
    plt.subplot(236), plt.title("(6) cv.add(img1, img3)"), plt.axis('off')
    plt.imshow(cv.cvtColor(imgAddCV, cv.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()