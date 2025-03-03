"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

#【1403】边缘检测之 DoG 算子（高斯差分算子）
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv.imread("../images/Fig1401.png", flags=0)  # 灰度图像

    # (1) 高斯核低通滤波器，sigmaY 缺省时 sigmaY=sigmaX
    GaussBlur1 = cv.GaussianBlur(img, (0,0), sigmaX=1.0)  # sigma=1.0
    GaussBlur2 = cv.GaussianBlur(img, (0,0), sigmaX=2.0)  # sigma=2.0
    GaussBlur3 = cv.GaussianBlur(img, (0,0), sigmaX=4.0)  # sigma=4.0
    GaussBlur4 = cv.GaussianBlur(img, (0,0), sigmaX=8.0)  # sigma=8.0

    # (2) 高斯拉普拉斯算子 LoG
    GaussLap1 = cv.Laplacian(GaussBlur2, cv.CV_32F, ksize=3)
    GaussLap2 = cv.Laplacian(GaussBlur3, cv.CV_32F, ksize=3)
    GaussLap3 = cv.Laplacian(GaussBlur4, cv.CV_32F, ksize=3)
    imgLoG1 = np.uint8(cv.normalize(GaussLap1, None, 0, 255, cv.NORM_MINMAX))
    imgLoG2 = np.uint8(cv.normalize(GaussLap2, None, 0, 255, cv.NORM_MINMAX))
    imgLoG3 = np.uint8(cv.normalize(GaussLap3, None, 0, 255, cv.NORM_MINMAX))

    # (3) 高斯差分算子 (DoG)
    imgDoG1 = cv.subtract(GaussBlur2, GaussBlur1)  # s2/s1=2.0/1.0
    imgDoG2 = cv.subtract(GaussBlur3, GaussBlur2)  # s3/s2=4.0/2.0
    imgDoG3 = cv.subtract(GaussBlur4, GaussBlur3)  # s4/s3=8.0/4.0

    plt.figure(figsize=(9, 6))
    plt.subplot(231), plt.title("(1) LoG (sigma=2.0)")
    plt.axis('off'), plt.imshow(imgLoG1, cmap='gray')
    plt.subplot(232), plt.title("(2) LoG (sigma=4.0)")
    plt.axis('off'), plt.imshow(imgLoG2, cmap='gray')
    plt.subplot(233), plt.title("(3) LoG (sigma=8.0)")
    plt.axis('off'), plt.imshow(imgLoG3, cmap='gray')
    plt.subplot(234), plt.title("(4) DoG (s2/s1=2.0/1.0)")
    plt.axis('off'), plt.imshow(imgDoG1, cmap='gray')
    plt.subplot(235), plt.title("(5) DoG (s2/s1=4.0/2.0)")
    plt.axis('off'), plt.imshow(imgDoG2, cmap='gray')
    plt.subplot(236), plt.title("(6) DoG (s2/s1=8.0/4.0)")
    plt.axis('off'), plt.imshow(imgDoG3, cmap='gray')
    plt.tight_layout()
    plt.show()

