"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【0208】LUT 查表实现颜色缩减
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    filepath = "../images/Lena.tif"  # 读取文件的路径
    gray = cv.imread(filepath, flags=0)  # flags=0 读取灰度图像
    h, w = gray.shape[:2]  # 图片的高度, 宽度

    timeBegin = cv.getTickCount()
    imgGray32 = np.empty((w,h), np.uint8)  # 创建空白数组
    for i in range(h):
        for j in range(w):
            imgGray32[i][j] = (gray[i][j]//8) * 8
    timeEnd = cv.getTickCount()
    time = (timeEnd-timeBegin)/cv.getTickFrequency()
    print("Grayscale reduction by nested loop: {} sec".format(round(time, 4)))

    timeBegin = cv.getTickCount()
    table32 = np.array([(i//8)*8 for i in range(256)]).astype(np.uint8)  # (256,)
    gray32 = cv.LUT(gray, table32)
    timeEnd = cv.getTickCount()
    time = (timeEnd-timeBegin)/cv.getTickFrequency()
    print("Grayscale reduction by cv.LUT: {} sec".format(round(time, 4)))

    table8 = np.array([(i//32)*32 for i in range(256)]).astype(np.uint8)  # (256,)
    gray8 = cv.LUT(gray, table8)

    plt.figure(figsize=(9, 3.5))
    plt.subplot(131), plt.axis('off'), plt.title("(1) Gray-256")
    plt.imshow(gray, cmap='gray')
    plt.subplot(132), plt.axis('off'), plt.title("(2) Gray-32")
    plt.imshow(gray32, cmap='gray')
    plt.subplot(133), plt.axis('off'), plt.title("(3) Gray-8")
    plt.imshow(gray8, cmap='gray')
    plt.tight_layout()
    plt.show()
