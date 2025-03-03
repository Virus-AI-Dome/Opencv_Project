"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【1202】形态学运算之开运算与闭运算
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv.imread("../images/Fig1201.png", flags=0)
    _, imgBin = cv.threshold(img, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)  # 二值处理

    # 图像腐蚀
    ksize = (5, 5)  # 结构元尺寸
    element = cv.getStructuringElement(cv.MORPH_RECT, ksize)  # 矩形结构元
    imgErode = cv.erode(imgBin, kernel=element)  # 腐蚀
    # 对腐蚀图像进行膨胀
    imgDilateErode = cv.dilate(imgErode, kernel=element)  # 腐蚀-膨胀
    # 图像的开运算
    imgOpen = cv.morphologyEx(imgBin, cv.MORPH_OPEN, kernel=element)

    # 图像膨胀
    ksize = (5, 5)  # 结构元尺寸
    element = cv.getStructuringElement(cv.MORPH_RECT, ksize)  # 矩形结构元
    imgDilate = cv.dilate(imgBin, kernel=element)  # 膨胀
    # 对膨胀图像进行腐蚀
    imgErodeDilate = cv.erode(imgDilate, kernel=element)  # 膨胀-腐蚀
    # 图像的闭运算
    imgClose = cv.morphologyEx(imgBin, cv.MORPH_CLOSE, kernel=element)

    plt.figure(figsize=(9, 5))
    plt.subplot(241), plt.axis('off'), plt.title("(1) Original")
    plt.imshow(imgBin, cmap='gray', vmin=0, vmax=255)
    plt.subplot(242), plt.title("(2) Eroded"), plt.axis('off')
    plt.imshow(imgErode, cmap='gray')
    plt.subplot(243), plt.axis('off'), plt.title("(3) Eroded & dilated")
    plt.imshow(imgDilateErode, cmap='gray')
    plt.subplot(244), plt.title("(4) Opening"), plt.axis('off')
    plt.imshow(imgOpen, cmap='gray')
    plt.subplot(245), plt.axis('off'), plt.title("(5) BinaryInv")
    plt.imshow(cv.bitwise_not(imgBin), cmap='gray', vmin=0, vmax=255)
    plt.subplot(246), plt.title("(6) Dilated"), plt.axis('off')
    plt.imshow(imgDilate, cmap='gray')
    plt.subplot(247), plt.axis('off'), plt.title("(7) Dilated & eroded")
    plt.imshow(imgErodeDilate, cmap='gray')
    plt.subplot(248), plt.title("(8) Closing"), plt.axis('off')
    plt.imshow(imgClose, cmap='gray')
    plt.tight_layout()
    plt.show()
