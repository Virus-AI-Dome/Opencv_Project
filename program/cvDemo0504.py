"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【0504】图像的乘法和除法
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv.imread("../images/Lena.tif")  # 读取彩色图像(BGR)

    # (1) Numpy 乘法和除法运算
    timeBegin = cv.getTickCount()
    scalar = np.array([[0.5, 1.5, 3.5]])  # 标量数组的值可以不同
    multiplyNP = img * scalar
    divideNP = img / scalar
    timeEnd = cv.getTickCount()
    time = (timeEnd - timeBegin) / cv.getTickFrequency()
    print("(1) Multiply by Numpy: {} sec".format(round(time, 4)))
    print("max(multiplyNP)={}, mean(multiplyNP)={:.1f}".format(multiplyNP.max(), multiplyNP.mean()))

    # (2) OpenCV 乘法和除法运算
    timeBegin = cv.getTickCount()
    scalar = np.array([[0.5, 1.5, 3.5]])  # 标量数组的值可以不同
    multiplyCV = cv.multiply(img, scalar)
    divideCV = cv.divide(img, scalar)
    timeEnd = cv.getTickCount()
    time = (timeEnd - timeBegin) / cv.getTickFrequency()
    print("(2) Multiply by OpenCV: {} sec".format(round(time, 4)))
    print("max(multiplyCV)={}, mean(multiplyCV)={:.1f}".format(multiplyCV.max(), multiplyCV.mean()))

    # (3) OpenCV 乘除法的标量与常数
    value = 1.5  # 常数，用于多通道图像时容易发生错误
    scalar = np.array([[1.5, 1.5, 1.5]])  # 推荐方法，标量是 (1,3) 数组
    multiplyCV1 = cv.multiply(img, scalar)
    multiplyCV2 = cv.multiply(img, value)
    print("(3) Difference between value and scalar:")
    print("mean(multiplyCV1)={:.1f}, mean(multiplyCV2)={:.1f}".format(multiplyCV1.mean(), multiplyCV2.mean()))
