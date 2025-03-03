"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【0501】图像的加法运算
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img1 = cv.imread("../images/Lena.tif")  # 读取彩色图像(BGR)
    img2 = cv.imread("../images/Fig0301.png")  # 读取彩色图像(BGR)
    h, w = img1.shape[:2]
    img3 = cv.resize(img2, (w,h))  # 调整图像大小与 img1 相同
    gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)  # 灰度图像
    print(img1.shape, img2.shape, img3.shape, gray.shape)

    # (1) 图像与常数相加
    value = 100  # 常数
    imgAddV = cv.add(img1, value)  # OpenCV 加法: 图像 + 常数
    imgAddG = cv.add(gray, value)  # OpenCV 加法: 灰度图像 + 常数
    # (2) 彩色图像与标量相加
    # scalar = (30, 40, 50, 60)  # 标量的函数定义是 4 个元素的元组
    scalar = np.ones((1, 3)) * value  # 推荐方法，标量为 (1, 3) 数组
    # scalar = np.array([[40, 50, 60]])  # 标量数组的值可以不同
    imgAddS = cv.add(img1, scalar)  # OpenCV 加法: 图像 + 标量
    # (3) Numpy 取模加法
    imgAddNP = img1 + img3  # # Numpy 加法: 模运算
    # (4) OpenCV 饱和加法
    # imgAddCV = cv.add(img1, img2)  # 错误：大小不同的图像不能用 cv.add 相加
    imgAddCV = cv.add(img1, img3)  # OpenCV 加法: 饱和运算

    plt.figure(figsize=(9, 6))
    plt.subplot(231), plt.title("(1) img1"), plt.axis('off')
    plt.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
    plt.subplot(232), plt.title("(2) add(img, value)"), plt.axis('off')
    plt.imshow(cv.cvtColor(imgAddV, cv.COLOR_BGR2RGB))
    plt.subplot(233), plt.title("(3) add(img, scalar)"), plt.axis('off')
    plt.imshow(cv.cvtColor(imgAddS, cv.COLOR_BGR2RGB))
    plt.subplot(234), plt.title("(4) img3"), plt.axis('off')
    plt.imshow(cv.cvtColor(img2, cv.COLOR_BGR2RGB))
    plt.subplot(235), plt.title("(5) img1 + img3"), plt.axis('off')
    plt.imshow(cv.cvtColor(imgAddNP, cv.COLOR_BGR2RGB))
    plt.subplot(236), plt.title("(6) cv.add(img1, img3)"), plt.axis('off')
    plt.imshow(cv.cvtColor(imgAddCV, cv.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()


