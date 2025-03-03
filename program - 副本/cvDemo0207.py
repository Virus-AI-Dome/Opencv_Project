"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【0207】LUT 查表实现图像反转
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    filepath = "../images/Lena.tif"  # 读取文件的路径
    img = cv.imread(filepath, flags=1)  # flags=1 读取彩色图像(BGR)
    h, w, ch = img.shape  # 图片的高度, 宽度 和通道数

    timeBegin = cv.getTickCount()
    imgInv = np.empty((w, h, ch), np.uint8)  # 创建空白数组
    for i in range(h):
        for j in range(w):
            for k in range(ch):
                imgInv[i][j][k] = 255 - img[i][j][k]
    timeEnd = cv.getTickCount()
    time = (timeEnd - timeBegin) / cv.getTickFrequency()
    print("Image invert by nested loop: {} sec".format(round(time, 4)))

    timeBegin = cv.getTickCount()
    transTable = np.array([(255 - i) for i in range(256)]).astype(np.uint8)  # (256,)
    invLUT = cv.LUT(img, transTable)
    timeEnd = cv.getTickCount()
    time = (timeEnd - timeBegin) / cv.getTickFrequency()
    print("Image invert by cv.LUT: {} sec".format(round(time, 4)))

    timeBegin = cv.getTickCount()
    subtract = 255 - img
    timeEnd = cv.getTickCount()
    time = (timeEnd - timeBegin) / cv.getTickFrequency()
    print("Image invert by subtraction: {} sec".format(round(time, 4)))
