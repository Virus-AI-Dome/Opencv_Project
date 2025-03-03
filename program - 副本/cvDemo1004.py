"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【1004】空间滤波之中值滤波器
import cv2 as cv
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv.imread("../images/Fig1002.png", flags=0)  # 读取灰度图像

    # (1) 高斯低通滤波核
    ksize = (11,11)
    GaussBlur = cv.GaussianBlur(img, ksize, 0, 0)
    # (2) 中值滤波器
    medianBlur = cv.medianBlur(img, ksize=3)

    plt.figure(figsize=(9, 3.5))
    plt.subplot(131), plt.axis('off'), plt.title("(1) Original")
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.subplot(132), plt.axis('off'), plt.title("(2) GaussianFilter")
    plt.imshow(GaussBlur, cmap='gray', vmin=0, vmax=255)
    plt.subplot(133), plt.axis('off'), plt.title("(3) MedianBlur(size=3)")
    plt.imshow(medianBlur, cmap='gray', vmin=0, vmax=255)
    plt.tight_layout()
    plt.show()
