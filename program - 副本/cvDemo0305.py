"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【0305】用多通道查找表调节色彩平衡
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv.imread("../images/Lena.tif", flags=1)  # 读取彩色

    # 生成单通道查找表，形状为 (256,)
    maxG = 128  # 修改颜色通道最大值，0<=maxG<=255
    lutHalf = np.array([int(i * maxG / 255) for i in range(256)]).astype("uint8")
    lutEqual = np.array([i for i in range(256)]).astype("uint8")
    # 构造多通道查找表，形状为 (1,256,3)
    lut3HalfB = np.dstack((lutHalf, lutEqual, lutEqual))  # B 通道衰减
    lut3HalfG = np.dstack((lutEqual, lutHalf, lutEqual))  # G 通道衰减
    lut3HalfR = np.dstack((lutEqual, lutEqual, lutHalf))  # R 通道衰减
    # 用多通道查找表进行颜色替换
    blendHalfB = cv.LUT(img, lut3HalfB)  # B 通道衰减 50%
    blendHalfG = cv.LUT(img, lut3HalfG)  # G 通道衰减 50%
    blendHalfR = cv.LUT(img, lut3HalfR)  # R 通道衰减 50%
    print(img.shape, blendHalfB.shape, lutHalf.shape, lut3HalfB.shape)

    plt.figure(figsize=(9, 3.5))
    plt.subplot(131), plt.axis('off'), plt.title("(1) B_ch half decayed")
    plt.imshow(cv.cvtColor(blendHalfB, cv.COLOR_BGR2RGB))
    plt.subplot(132), plt.axis('off'), plt.title("(2) G_ch half decayed")
    plt.imshow(cv.cvtColor(blendHalfG, cv.COLOR_BGR2RGB))
    plt.subplot(133), plt.axis('off'), plt.title("(3) R_ch half decayed")
    plt.imshow(cv.cvtColor(blendHalfR, cv.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()
