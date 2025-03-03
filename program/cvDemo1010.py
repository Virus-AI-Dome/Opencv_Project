"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【1010】空间滤波之 Laplacian 算子
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv.imread("../images/Fig1001.png", flags=0)
    print(img.shape[:2])

    # (1) 使用函数 filter2D 计算 Laplace 算子 K1，K2
    LapLacianK1 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])  # Laplacian K1
    imgLapK1 = cv.filter2D(img, -1, LapLacianK1, cv.BORDER_REFLECT)
    LapLacianK2 = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])  # Laplacian K2
    imgLapK2 = cv.filter2D(img, -1, LapLacianK2, cv.BORDER_REFLECT)

    # (2) 使用 cv.Laplacian 计算 Laplace 算子
    imgLaplacian = cv.Laplacian(img, cv.CV_32F, ksize=3)  # 输出为浮点型
    absLaplacian = cv.convertScaleAbs(imgLaplacian)  # 拉伸到 [0,255]
    print(type(imgLaplacian[0,0]), type(absLaplacian[0,0]))

    plt.figure(figsize=(9, 3.2))
    plt.subplot(131), plt.axis('off'), plt.title("(1) Original")
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.subplot(132), plt.axis('off'), plt.title("(2) Laplacian(float)")
    plt.imshow(imgLaplacian, cmap='gray', vmin=0, vmax=255)
    plt.subplot(133), plt.axis('off'), plt.title("(3) Laplacian(abs)")
    plt.imshow(absLaplacian, cmap='gray', vmin=0, vmax=255)
    plt.tight_layout()
    plt.show()

