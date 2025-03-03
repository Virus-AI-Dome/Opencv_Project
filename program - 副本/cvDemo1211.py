"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【1211】形态学算法之线条细化算法
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def linesThinning(image):
    # 背景为白色(255)，被细化物体为黑色(0)
    array = [0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, \
             1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, \
             0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, \
             1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, \
             1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
             1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, \
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
             0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, \
             1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, \
             0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, \
             1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, \
             1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
             1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, \
             1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, \
             1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0]

    h, w = image.shape[0], image.shape[1]
    imgThin = image.copy()
    for i in range(h):
        for j in range(w):
            if image[i, j] == 0:
                a = np.ones((9,), np.int16)  # np.int
                for k in range(3):
                    for l in range(3):
                        if -1<(i-1+k)< h and -1<(j-1+l)<w and imgThin[i-1+k, j-1+l] == 0:
                            a[k*3+l] = 0
                sum = a[0]*1 + a[1]*2 + a[2]*4 + a[3]*8 + a[5]*16 + a[6]*32 + a[7]*64 + a[8]*128
                imgThin[i, j] = array[sum] * 255
    return imgThin

if __name__ == '__main__':
    img = cv.imread("../images/Fig1202.png", flags=0)  # 读取为灰度图像

    # (1) 阈值处理后细化处理
    _, binary = cv.threshold(img, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)  # 二值处理
    imgThin = linesThinning(binary)  # 细化算法

    # (2) 孔洞填充后细化处理
    element = cv.getStructuringElement(cv.MORPH_RECT, (7,7))  # 矩形结构元
    imgOpen = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel=element)  # 填充孔洞
    imgThin2 = linesThinning(imgOpen)  # 细化算法

    plt.figure(figsize=(9, 3.2))
    plt.subplot(131), plt.axis('off'), plt.title("(1) Original")
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.subplot(132), plt.axis('off'), plt.title("(2) Thinned with holes")
    plt.imshow(imgThin, cmap='gray', vmin=0, vmax=255)
    plt.subplot(133), plt.axis('off'), plt.title("(3) Thinned lines")
    plt.imshow(imgThin2, cmap='gray', vmin=0, vmax=255)
    plt.tight_layout()
    plt.show()

