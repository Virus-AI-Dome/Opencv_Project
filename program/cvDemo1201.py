"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【1201】形态学运算之腐蚀与膨胀
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv.imread("../images/Fig1201.png", flags=0)
    _, imgBin = cv.threshold(img, 20, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)  # 二值处理

    # 图像腐蚀
    ksize1 = (3, 3)  # 结构元尺寸 3*3
    kernel1 = np.ones(ksize1, dtype=np.uint8)  # 矩形结构元
    imgErode1 = cv.erode(imgBin, kernel=kernel1)  # 图像腐蚀
    kernel2 = np.ones((9, 9), dtype=np.uint8)
    imgErode2 = cv.erode(imgBin, kernel=kernel2)
    imgErode3 = cv.erode(imgBin, kernel=kernel1, iterations=2)  # 腐蚀 2 次
    # 图像膨胀
    ksize1 = (3, 3)  # 结构元尺寸 3*3
    kernel1 = cv.getStructuringElement(cv.MORPH_RECT, ksize1)  # 矩形结构元
    imgDilate1 = cv.dilate(imgBin, kernel=kernel1)  # 图像膨胀
    kernel2 = cv.getStructuringElement(cv.MORPH_RECT, (9, 9))  # 矩形结构元
    imgDilate2 = cv.dilate(imgBin, kernel=kernel2)
    imgDilate3 = cv.dilate(imgBin, kernel=kernel1, iterations=3)  # 膨胀 2次
    # 对腐蚀图像进行膨胀
    dilateErode = cv.dilate(imgErode2, kernel=kernel2)  # 图像膨胀

    plt.figure(figsize=(9, 5))
    plt.subplot(241), plt.axis('off'), plt.title("(1) Original")
    plt.imshow(imgBin, cmap='gray', vmin=0, vmax=255)
    plt.subplot(242), plt.title("(2) Eroded size=(3,3)"), plt.axis('off')
    plt.imshow(imgErode1, cmap='gray')
    plt.subplot(243), plt.title("(3) Eroded size=(9,9)"), plt.axis('off')
    plt.imshow(imgErode2, cmap='gray')
    plt.subplot(244), plt.title("(4) Eroded size=(3,3)*2"), plt.axis('off')
    plt.imshow(imgErode3, cmap='gray')
    plt.subplot(245), plt.title("(5) Eroded & Dilated"), plt.axis('off')
    plt.imshow(dilateErode, cmap='gray')
    plt.subplot(246), plt.title("(6) Dilated size=(3,3)"), plt.axis('off')
    plt.imshow(imgDilate1, cmap='gray')
    plt.subplot(247), plt.title("(7) Dilated size=(9,9)"), plt.axis('off')
    plt.imshow(imgDilate2, cmap='gray')
    plt.subplot(248), plt.title("(8) Dilated size=(3,3)*2"), plt.axis('off')
    plt.imshow(imgDilate3, cmap='gray')
    plt.tight_layout()
    plt.show()
