"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""
import cv2
# 【0305】用多通道查找表调节色彩平衡
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv.imread("../images/Lena.tif", flags=1)  # 读取彩色

    maxG = 128
    #  numpy 数组，每个值都通过 maxG / 255 缩放以适应 0-128 的范围。
    lutHalf = np.array([int(i * maxG / 255) for i in range(256)]).astype("uint8")
    lutEqual = np.array([i for i in range(256)]).astype("uint8")

    lut3HalfB = np.dstack((lutHalf, lutEqual, lutEqual))
    lut3HalfG = np.dstack((lutEqual, lutHalf, lutEqual))
    lut3HalfR = np.dstack((lutEqual, lutEqual, lutHalf))

    print(lut3HalfG)

    blendHalfB = cv.LUT(img, lut3HalfB)
    blendHalfG = cv.LUT(img, lut3HalfG)
    blendHalfR = cv.LUT(img, lut3HalfR)

    images = [blendHalfB, blendHalfG, blendHalfR]
    titles = ['B', 'G', 'R']
    plt.figure(figsize=(9, 3.5))
    for i in range(len(images)):
        plt.subplot(1, 3, i + 1)
        plt.axis('off')
        plt.title('({}) {}_Ch half decayed'.format(i,titles[i]))
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()




