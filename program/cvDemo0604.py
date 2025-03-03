"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【0604】图像的翻转 (镜像操作)
import cv2 as cv
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv.imread("../images/Fig0301.png")  # 读取彩色图像(BGR)
    imgFlipH = cv.flip(img, 0)
    imgFlipV = cv.flip(img, 1)
    imgFlipHV = cv.flip(img, -1)
    plt.figure(figsize=(7, 5))
    plt.subplot(221), plt.axis('off'), plt.title("(1) Original")
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))  # 原始图像
    plt.subplot(222), plt.axis('off'), plt.title("(2) Flip Horizontally")
    plt.imshow(cv.cvtColor(imgFlipH, cv.COLOR_BGR2RGB))  # 水平翻转
    plt.subplot(223), plt.axis('off'), plt.title("(3) Flip Vertically")
    plt.imshow(cv.cvtColor(imgFlipV, cv.COLOR_BGR2RGB))  # 垂直翻转
    plt.subplot(224), plt.axis('off'), plt.title("(4) Flipped Hori&Vert")
    plt.imshow(cv.cvtColor(imgFlipHV, cv.COLOR_BGR2RGB))  # 水平垂直翻转
    plt.tight_layout()
    plt.show()





