"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【0601】图像的平移
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv.imread("../images/Lena.tif")
    height, width = img.shape[:2]

    dx , dy = 100, 50
    MAT = np.float32([[1, 0, dx], [0, 1, dy]])

    imgTrans1 = cv.warpAffine(img, MAT, (width, height), borderValue=(255, 255, 255))
    imgTrans2 = cv.warpAffine(img, MAT, (601, 401), borderValue=(255, 255, 255))

    plt.figure(figsize=(9, 3.5))
    plt.subplot(1, 3, 1)
    plt.title('Original')

    plt.imshow(cv.cvtColor(img,cv.COLOR_BGR2RGB))
    plt.subplot(1, 3, 2)
    plt.title('Affine1')

    plt.imshow(cv.cvtColor(imgTrans1, cv.COLOR_BGR2RGB))
    plt.subplot(1, 3, 3)
    plt.title('Affine2')

    plt.imshow(cv.cvtColor(imgTrans2, cv.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()