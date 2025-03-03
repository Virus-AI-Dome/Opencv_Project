"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【0705】灰度变换之幂律变换 (伽马变换)
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    gray = cv.imread("../images/Fig0701.png", flags=0)

    c = 1
    gammas = [0.25, 0.50, 1.0, 1.5, 2.0, 4.0]
    fig = plt.figure(figsize=(10, 5))
    for i in range(len(gammas)):
        ax = fig.add_subplot(2, 3, i + 1)
        img_gamma = c * np.power(gray, gammas[i])     # np.power 为幂运算
        ax.imshow(img_gamma,cmap='gray')
        if gammas[i] == 1.0:
            ax.set_title("(3) original")
        else:
            ax.set_title(f"{i+1} gammas={gammas[i]}")
    plt.tight_layout()
    plt.show()


