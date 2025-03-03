"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【0707】分段线性变换之灰度级分层
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    gray = cv.imread("../images/Fig0703.png", flags=0)  # 读取为灰度图像
    width, height = gray.shape[:2]

    a, b = 155, 245
    # 1, 二值变换灰度级分成
    binLayer = gray.copy()
    binLayer[(binLayer[:, :] < a) | (binLayer[:, :] > b)] = 0
    binLayer[(binLayer[:, :] >= a) & (binLayer[:, :] <= b)] = 245

    # 2, 增强选择的灰度窗口
    a, b = 155, 245
    winLayer = gray.copy()
    winLayer[(winLayer[:, :] >= a) & (winLayer[:, :] <= b)] = 245

    plt.figure(figsize=(9, 3.5))
    plt.subplot(131), plt.axis('off'), plt.title("(1) Original")
    plt.imshow(gray, cmap='gray', vmin=0, vmax=255)
    plt.subplot(132), plt.axis('off'), plt.title("(2) Binary layered")
    plt.imshow(binLayer, cmap='gray', vmin=0, vmax=255)
    plt.subplot(133), plt.axis('off'), plt.title("(3) Window layered")
    plt.imshow(winLayer, cmap='gray', vmin=0, vmax=255)
    plt.tight_layout()
    plt.show()
