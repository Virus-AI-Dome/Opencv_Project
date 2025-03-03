"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【0302】灰度图像转换为伪彩色图像
import cv2 as cv
from matplotlib import pyplot as plt

if __name__ == '__main__':
    # 读取原始图像
    gray = cv.imread("../images/Fig0301.png", flags=0)  # 读取灰度图像
    h, w = gray.shape[:2]  # 图片的高度, 宽度

    # 伪彩色处理
    pseudo1 = cv.applyColorMap(gray, colormap=cv.COLORMAP_HOT)
    pseudo2 = cv.applyColorMap(gray, colormap=cv.COLORMAP_PINK)
    pseudo3 = cv.applyColorMap(gray, colormap=cv.COLORMAP_RAINBOW)
    pseudo4 = cv.applyColorMap(gray, colormap=cv.COLORMAP_HSV)
    pseudo5 = cv.applyColorMap(gray, colormap=cv.COLORMAP_TURBO)

    plt.figure(figsize=(9, 6))
    plt.subplot(231), plt.axis('off'), plt.title("(1) GRAY"), plt.imshow(gray, cmap='gray')
    plt.subplot(232), plt.axis('off'), plt.title("(2) COLORMAP_HOT")
    plt.imshow(cv.cvtColor(pseudo1, cv.COLOR_BGR2RGB))
    plt.subplot(233), plt.axis('off'), plt.title("(3) COLORMAP_PINK")
    plt.imshow(cv.cvtColor(pseudo2, cv.COLOR_BGR2RGB))
    plt.subplot(234), plt.axis('off'), plt.title("(4) COLORMAP_RAINBOW")
    plt.imshow(cv.cvtColor(pseudo3, cv.COLOR_BGR2RGB))
    plt.subplot(235), plt.axis('off'), plt.title("(5) COLORMAP_HSV")
    plt.imshow(cv.cvtColor(pseudo4, cv.COLOR_BGR2RGB))
    plt.subplot(236), plt.axis('off'), plt.title("(6) COLORMAP_TURBO")
    plt.imshow(cv.cvtColor(pseudo5, cv.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()



