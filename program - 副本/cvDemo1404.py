"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

#【1404】边缘检测之 Canny 算子
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv.imread("../images/Fig1401.png", flags=0)  # 灰度图像

    # (1) 高斯差分算子 DoG
    GaussBlur1 = cv.GaussianBlur(img, (0,0), sigmaX=1.0)  # sigma=1.0
    GaussBlur2 = cv.GaussianBlur(img, (0,0), sigmaX=2.0)  # sigma=2.0
    imgDoG1 = cv.subtract(GaussBlur2, GaussBlur1)  # s2/s1=2.0/1.0

    # (2) 高斯拉普拉斯算子 LoG
    GaussLap1 = cv.Laplacian(GaussBlur1, cv.CV_32F, ksize=3)
    imgLoG1 = np.uint8(cv.normalize(GaussLap1, None, 0, 255, cv.NORM_MINMAX))

    # (3) Canny 边缘检测算子
    TL, TH = 50, 150
    imgCanny = cv.Canny(img, TL, TH)

    plt.figure(figsize=(9, 3.3))
    plt.subplot(131), plt.title("(1) LoG (sigma=2.0)")
    plt.axis('off'), plt.imshow(imgLoG1, cmap='gray')
    plt.subplot(132), plt.title("(2) DoG (s2/s1=2.0/1.0)")
    plt.axis('off'), plt.imshow(imgDoG1, cmap='gray')
    plt.subplot(133), plt.title("(3) Canny (TH/TL=150/50)")
    plt.axis('off'), plt.imshow(imgCanny, cmap='gray')
    plt.tight_layout()
    plt.show()
