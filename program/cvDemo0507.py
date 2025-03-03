"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【0507】基于积分图像的均值滤波器
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    # 读取目标文件
    img = cv.imread("../images/Lena.tif", flags=0)
    H, W = img.shape[:2]
    k = 15
    # (1) 两重循环实现均值滤波
    timeBegin = cv.getTickCount()
    pad = k//2 + 1
    imgPad = cv.copyMakeBorder(img, pad, pad, pad, pad, cv.BORDER_REFLECT)
    imgBox1 = np.zeros((H, W), np.int32)
    for h in range(H):
        for w in range(W):
            # 这行代码的作用是对图像 imgPad 的一个局部区域（大小为 k × k）进行求和，并计算该区域的 均值。
            # 主要用于 均值滤波（Mean Filter），也就是 平滑图像、去除噪声
            imgBox1[h, w] = np.sum(imgPad[h:h+k, w:w+k]) / (k * k)
    timeEnd = cv.getTickCount()
    print("Blurred by double cycle: {} sec".format(round(timeEnd - timeBegin), 4))

    # (2) 基于blue实现均值滤波
    timeBegin = cv.getTickCount()
    imgBlur =  img.copy()
    imgBlur = cv.blur(imgBlur, (15, 15))
    timeEnd = cv.getTickCount()
    print("Blurred by single cycle: {} sec".format(round(timeEnd - timeBegin), 4))

    # (3) 基于 cv.boxFilter
    timeBegin = cv.getTickCount()

    imgBoxf = cv.boxFilter(img, -1, (15, 15))
    timeEnd = cv.getTickCount()
    print("Boxf: {} sec".format(round(timeEnd - timeBegin), 4))

    plt.figure(figsize=(9, 6))
    plt.subplot(131), plt.axis('off'), plt.title("imgBlur image")
    plt.imshow(imgBlur, cmap='gray', vmin=0, vmax=255)
    plt.subplot(132), plt.axis('off'), plt.title("imgBox1 by dual cycle")
    plt.imshow(imgBox1, cmap='gray', vmin=0, vmax=255)
    plt.subplot(133), plt.axis('off'), plt.title("imgBoxf by integral image")
    plt.imshow(imgBoxf, cmap='gray', vmin=0, vmax=255)
    plt.tight_layout()
    plt.show()







