"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""


# 【1607】特征描述之圆形扩展 LBP 描述符
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def basicLBP(gray):
    height, width = gray.shape
    dst = np.zeros((height, width), np.uint8)
    kernelFlatten = np.array([1, 2, 4, 128, 0, 8, 64, 32, 16])  # 从左上角开始顺时针旋转
    for h in range(1, height-1):
        for w in range(1, width-1):
            LBPFlatten = (gray[h-1:h+2, w-1:w+2] >= gray[h, w]).flatten()  # 展平为一维向量, (9,)
            dst[h, w] = np.vdot(LBPFlatten, kernelFlatten)  # 一维向量的内积
    return dst

# extend LBP，在半径为 R 的圆形邻域内有 N 个采样点
def extendLBP(gray, r=3, n=8):
    height, width = gray.shape
    ww = np.empty((n, 4), np.float64)  # (8,4)
    p = np.empty((n, 4), np.int16)  # [x1, y1, x2, y2]
    for k in range(n):  # 双线性插值估计坐标偏移量和权值
        # 计算坐标偏移量 rx，ry
        rx = r * np.cos(2.0 * np.pi * k / n)
        ry = -(r * np.sin(2.0 * np.pi * k / n))
        # 对采样点分别进行上下取整
        x1, y1 = int(np.floor(rx)), int(np.floor(ry))
        x2, y2 = int(np.ceil(rx)), int(np.ceil(ry))
        # 将坐标偏移量映射到 0~1
        tx = rx - x1
        ty = ry - y1
        # 计算插值的权重
        ww[k, 0] = (1 - tx) * (1 - ty)
        ww[k, 1] = tx * (1 - ty)
        ww[k, 2] = (1 - tx) * ty
        ww[k, 3] = tx * ty
        p[k,0], p[k,1], p[k,2], p[k,3] = x1, y1, x2, y2

    dst = np.zeros((height-2*r, width-2*r), np.uint8)
    for h in range(r, height - r):
        for w in range(r, width - r):
            center = gray[h, w]  # 中心像素点的灰度值
            for k in range(n):
                # 双线性插值估计采样点 k 的灰度值
                # neighbor = gray[i+y1,j+x1]*w1 + gray[i+y2,j+x1]*w2 + gray[i+y1,j+x2]*w3 + gray[i+y2,j+x2]*w4
                x1, y1, x2, y2 = p[k,0], p[k,1], p[k,2], p[k,3]
                gInterp = np.array(
                    [gray[h+y1, w+x1], gray[h+y2, w+x1], gray[h+y1, w+x2], gray[h+y2, w+x2]])
                wFlatten = ww[k, :]
                grayNeighbor = np.vdot(gInterp, wFlatten)  # 一维向量的内积
                # 由 N 个采样点与中心像素点的灰度值比较，构造 LBP 特征编码
                dst[h-r, w-r] |= (grayNeighbor > center) << (np.uint8)(n-k-1)
    return dst

if __name__ == '__main__':
    img = cv.imread("../images/Fig1604.png", flags=1)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 灰度图像

    timeBegin = cv.getTickCount()
    imgLBP1 = basicLBP(gray)  # 从右上角开始顺时针旋转
    timeEnd = cv.getTickCount()
    time = (timeEnd - timeBegin) / cv.getTickFrequency()
    print("(1) basicLBP: {} sec".format(round(time, 4)))

    timeBegin = cv.getTickCount()
    r1, n1 = 3, 8
    imgLBP2 = extendLBP(gray, r1, n1)
    timeEnd = cv.getTickCount()
    time = (timeEnd - timeBegin) / cv.getTickFrequency()
    print("(2) extendLBP(r={},n={}): {} sec".format(r1, n1, round(time, 4)))

    timeBegin = cv.getTickCount()
    r2, n2 = 5, 8
    imgLBP3 = extendLBP(gray, r2, n2)
    timeEnd = cv.getTickCount()
    time = (timeEnd - timeBegin) / cv.getTickFrequency()
    print("(3) extendLBP(r={},n={}): {} sec".format(r2, n2, round(time, 4)))

    plt.figure(figsize=(9, 3.3))
    plt.subplot(131), plt.axis('off'), plt.title("(1) Basic LBP")
    plt.imshow(imgLBP1, 'gray')
    plt.subplot(132), plt.title("(2) Extend LBP (r={},n={})".format(r1, n1))
    plt.imshow(imgLBP2, 'gray'), plt.axis('off')
    plt.subplot(133), plt.title("(3) Extend LBP (r={},n={})".format(r2, n2))
    plt.imshow(imgLBP3, 'gray'), plt.axis('off')
    plt.tight_layout()
    plt.show()

