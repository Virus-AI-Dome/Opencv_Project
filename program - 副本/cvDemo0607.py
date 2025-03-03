"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【0607】图像的重映射
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv.imread("../images/Fig0301.png")  # 读取彩色图像(BGR)
    height, width = img.shape[:2]  # (250, 300)

    mapx = np.zeros((height, width), np.float32)  # 初始化
    mapy = np.zeros((height, width), np.float32)
    for h in range(height):
        for w in range(width):
            mapx[h,w] = w  # 水平方向不变
            mapy[h,w] = h  # 垂直方向不变
    dst1 = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

    mapx = np.array([[i*1.5 for i in range(width)] for j in range(height)], dtype=np.float32)
    mapy = np.array([[j*1.5 for i in range(width)] for j in range(height)], dtype=np.float32)
    dst2 = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)  # 尺寸缩放

    mapx = np.array([[i for i in range(width)] for j in range(height)], dtype=np.float32)  # 行不变
    mapy = np.array([[j for i in range(width)] for j in range(height-1, -1, -1)], dtype=np.float32)
    dst3 = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)  # 上下翻转，x 不变 y 翻转

    mapx = np.array([[i for i in range(width-1, -1, -1)] for j in range(height)], dtype=np.float32)
    mapy = np.array([[j for i in range(width)] for j in range(height)], dtype=np.float32)
    dst4 = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)  # 左右翻转，x 翻转 y 不变

    mapx = np.array([[i for i in range(width-1, -1, -1)] for j in range(height)], dtype=np.float32)
    mapy = np.array([[j for i in range(width)] for j in range(height-1, -1, -1)], dtype=np.float32)
    dst5 = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)  # 水平和垂直翻转，x 翻转 y 翻转

    print(img.shape, mapx.shape, mapy.shape, dst1.shape)
    plt.figure(figsize=(9,6))
    plt.subplot(231), plt.title("(1) Original"), plt.axis('off')
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.subplot(232), plt.title("(2) Copy"), plt.axis('off')
    plt.imshow(cv.cvtColor(dst1, cv.COLOR_BGR2RGB))
    plt.subplot(233), plt.title("(3) Resize"), plt.axis('off')
    plt.imshow(cv.cvtColor(dst2, cv.COLOR_BGR2RGB))
    plt.subplot(234), plt.title("(4) Flip vertical"), plt.axis('off')
    plt.imshow(cv.cvtColor(dst3, cv.COLOR_BGR2RGB))
    plt.subplot(235), plt.title("(5) Flip horizontal"), plt.axis('off')
    plt.imshow(cv.cvtColor(dst4, cv.COLOR_BGR2RGB))
    plt.subplot(236), plt.title("(6) Flip horizontal"), plt.axis('off')
    plt.imshow(cv.cvtColor(dst5, cv.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()
