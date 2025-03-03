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
    img = cv.imread("../images/Lena.tif",flags=0)  # 读取灰度图像
    H, W = img.shape[:2]

    k = 15  # 均值滤波器的尺寸
    # (1) 两重循环实现均值滤波
    timeBegin = cv.getTickCount()
    pad = k//2 + 1
    imgPad = cv.copyMakeBorder(img, pad, pad, pad, pad, cv.BORDER_REFLECT)
    imgBox1 = np.zeros((H, W), np.int32)
    for h in range(H):
        for w in range(W):
            #这行代码的作用是对图像 imgPad 的一个局部区域（大小为 k × k）进行求和，并计算该区域的 均值。
            # 主要用于 均值滤波（Mean Filter），也就是 平滑图像、去除噪声
            imgBox1[h,w] = np.sum(imgPad[h:h+k, w:w+k]) / (k*k)
    timeEnd = cv.getTickCount()
    time = (timeEnd - timeBegin) / cv.getTickFrequency()
    print("Blurred by double cycle: {} sec".format(round(time, 4)))

    # (2) 基于积分图像实现均值滤波
    timeBegin = cv.getTickCount()
    pad = k//2 + 1
    imgPadded = cv.copyMakeBorder(img, pad, pad, pad, pad, cv.BORDER_REFLECT)
    sumImg = cv.integral(imgPadded)
    imgBox2 = np.zeros((H, W), np.uint8)
    imgBox2[:,:] = (sumImg[:H,:W] - sumImg[:H, k:W+k] - sumImg[k:H+k,:W] + sumImg[k:H+k, k:W+k]) / (k*k)
    timeEnd = cv.getTickCount()
    time = (timeEnd - timeBegin) / cv.getTickFrequency()
    print("Blurred by integral image: {} sec".format(round(time, 4)))

    # (3) cv.boxFilter 实现均值滤波
    timeBegin = cv.getTickCount()
    kernel = np.ones(k, np.float32) / (k * k)  # 生成归一化核
    imgBoxF = cv.boxFilter(img, -1, (k, k))  # cv.boxFilter
    timeEnd = cv.getTickCount()
    time = (timeEnd - timeBegin) / cv.getTickFrequency()
    print("Blurred by cv.boxFilter: {} sec".format(round(time, 4)))

    plt.figure(figsize=(9, 6))
    plt.subplot(141), plt.axis('off'), plt.title("Original image")
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.subplot(142), plt.axis('off'), plt.title("Blurred by dual cycle")
    plt.imshow(imgBox1, cmap='gray', vmin=0, vmax=255)
    plt.subplot(143), plt.axis('off'), plt.title("Blurred by integral image")
    plt.imshow(imgBox2, cmap='gray', vmin=0, vmax=255)
    plt.subplot(144), plt.axis('off'), plt.title("Blurred by integral image")
    plt.imshow(imgBoxF, cmap='gray', vmin=0, vmax=255)
    plt.tight_layout()
    plt.show()

