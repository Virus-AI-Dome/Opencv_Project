"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【0706】分段线性变换之对比度拉伸
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    gray = cv.imread("../images/Fig0703.png", flags=0)

    # 拉伸控制点
    r1, s1 = 128, 64  # 第一转折点 (r1,s1)
    r2, s2 = 192, 224  # 第二转折点 (r2,s2)

    # LUT 快速查表法实现对比度拉伸
    luTable = np.zeros(256)
    for i in range(256):
        if i < r1:
            luTable[i] = (s1/r1) * i
        elif i < r2:
            luTable[i] = ((s2-s1)/(r2-r1))*(i-r1) + s1
        else:
            luTable[i] = ((s2-255.0)/(r2-255.0))*(i-r2) + s2
    imgSLT = np.uint8(cv.LUT(gray, luTable))  # 转换为 CV_8U




    print(luTable)
    plt.figure(figsize=(9, 3))
    plt.subplot(131), plt.axis('off'), plt.title("(1) Original")
    plt.imshow(gray, cmap='gray', vmin=0, vmax=255)
    plt.subplot(132), plt.title("(2) s=T(r)")
    r = [0, r1, r2, 255]
    s = [0, s1, s2, 255]
    plt.plot(r, s)
    plt.axis([0, 256, 0, 256])
    plt.text(128, 40, "(r1,s1)", fontsize=10)
    plt.text(128, 220, "(r2,s2)", fontsize=10)
    plt.xlabel("r, Input value")
    plt.ylabel("s, Output value")
    plt.subplot(133), plt.axis('off'), plt.title("(3) Stretched")
    plt.imshow(imgSLT, cmap='gray', vmin=0, vmax=255)
    plt.tight_layout()
    plt.show()

