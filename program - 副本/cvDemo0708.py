"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【0708】分段线性变换之比特平面分层
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    gray = cv.imread("../images/Fig0703.png", flags=0)  # flags=0 读取为灰度图像
    height, width = gray.shape[:2]  # 图片的高度和宽度

    bitLayer = np.zeros((8, height, width), np.uint(8))
    bitLayer[0] = cv.bitwise_and(gray, 1)  # 按位与 00000001
    bitLayer[1] = cv.bitwise_and(gray, 2)  # 按位与 00000010
    bitLayer[2] = cv.bitwise_and(gray, 4)  # 按位与 00000100
    bitLayer[3] = cv.bitwise_and(gray, 8)  # 按位与 00001000
    bitLayer[4] = cv.bitwise_and(gray, 16)  # 按位与 00010000
    bitLayer[5] = cv.bitwise_and(gray, 32)  # 按位与 0010000
    bitLayer[6] = cv.bitwise_and(gray, 64)  # 按位与 0100000
    bitLayer[7] = cv.bitwise_and(gray, 128)  # 按位与 1000000

    plt.figure(figsize=(9, 8))
    plt.subplot(331), plt.axis('off'), plt.title("(1) Original")
    plt.imshow(gray, cmap='gray', vmin=0, vmax=255)
    for bit in range(8):
        plt.subplot(3,3,9-bit), plt.axis('off'), plt.title(f"({9-bit}) {bin((bit))}")
        plt.imshow(bitLayer[bit], cmap='gray')
    plt.tight_layout()
    plt.show()

