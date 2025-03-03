"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【0306】调节图像的饱和度与明度
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv.imread("../images/Lena.tif", flags=1)  # 读取彩色图像
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)  # 色彩空间转换，BGR->HSV

    # 生成单通道查找表，形状为 (256,)
    k = 0.6  # 用户设定的色彩拉伸系数
    lutWeaken = np.array([int(k * i) for i in range(256)]).astype("uint8")
    lutEqual = np.array([i for i in range(256)]).astype("uint8")
    lutRaisen = np.array([int(255*(1-k) + k*i) for i in range(256)]).astype("uint8")
    # 构造多通道查找表，调节饱和度
    lutSWeaken = np.dstack((lutEqual, lutWeaken, lutEqual))  # 饱和度衰减
    lutSRaisen = np.dstack((lutEqual, lutRaisen, lutEqual))  # 饱和度增强
    # 构造多通道查找表，调节明度
    lutVWeaken = np.dstack((lutEqual, lutEqual, lutWeaken))  # 明度衰减
    lutVRaisen = np.dstack((lutEqual, lutEqual, lutRaisen))  # 明度增强
    # 用多通道查找表进行颜色替换
    blendSWeaken = cv.LUT(hsv, lutSWeaken)  # 饱和度降低
    blendSRaisen = cv.LUT(hsv, lutSRaisen)  # 饱和度增大
    blendVWeaken = cv.LUT(hsv, lutVWeaken)  # 明度降低
    blendVRaisen = cv.LUT(hsv, lutVRaisen)  # 明度升高

    plt.figure(figsize=(9, 6))
    plt.subplot(231), plt.axis('off'), plt.title("(1) Saturation weaken")
    plt.imshow(cv.cvtColor(blendSWeaken, cv.COLOR_HSV2RGB))
    plt.subplot(232), plt.axis('off'), plt.title("(2) Original saturation")
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.subplot(233), plt.axis('off'), plt.title("(3) Saturation raisen")
    plt.imshow(cv.cvtColor(blendSRaisen, cv.COLOR_HSV2RGB))
    plt.subplot(234), plt.axis('off'), plt.title("(4) Value weaken")
    plt.imshow(cv.cvtColor(blendVWeaken, cv.COLOR_HSV2RGB))
    plt.subplot(235), plt.axis('off'), plt.title("(5) Original value")
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.subplot(236), plt.axis('off'), plt.title("(6) Value raisen")
    plt.imshow(cv.cvtColor(blendVRaisen, cv.COLOR_HSV2RGB))
    plt.tight_layout()
    plt.show()
