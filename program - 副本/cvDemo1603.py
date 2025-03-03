"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【1603】特征描述之傅里叶频谱分析
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def halfcircle(radius, x0, y0):  # 计算圆心(x0,y0) 半径为 r 的半圆的整数坐标
    degree = np.arange(180, 360, 1)  # 因对称性可以用半圆 (180,)
    theta = np.float32(degree * np.pi/180)  # 弧度，一维数组 (180,)
    xc = (x0 + radius * np.cos(theta)).astype(np.int16)  # 计算直角坐标，整数
    yc = (y0 + radius * np.sin(theta)).astype(np.int16)
    return xc, yc

def intline(x1, x2, y1, y2):  # 计算从(x1,y1)到(x2,y2)的线段上所有点的坐标
    dx, dy = np.abs(x2 - x1), np.abs(y2 - y1)  # x, y 的增量
    if dx == 0 and dy == 0:
        x, y = np.array([x1]), np.array([y1])
        return x, y
    if dx > dy:
        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
        m = (y2-y1) / (x2-x1)
        x = np.arange(x1, x2+1, 1)  # [x1,x2]
        y = (y1 + m*(x-x1)).astype(np.int16)
    else:
        if y1 > y2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
        m = (x2-x1) / (y2-y1)
        y = np.arange(y1, y2+1, 1)  # [y1,y2]
        x = (x1 + m*(y-y1)).astype(np.int16)
    return x, y

def specxture(gray):  # cv.dft 实现图像的傅里叶变换
    height, width = gray.shape
    x0, y0 = int(height / 2), int(width / 2)  # x0=300, y0=300
    rmax = min(height, width) // 2 - 1  # rmax=299
    # print(height, width, x0, y0, rmax)
    # FFT 变换
    gray32 = np.float32(gray)  # 将图像转换成 float32
    dft = cv.dft(gray32, flags=cv.DFT_COMPLEX_OUTPUT)  # 傅里叶变换，(600, 600, 2)
    dftShift = np.fft.fftshift(dft)  # 将低频分量移动到频域图像的中心
    sAmp = cv.magnitude(dftShift[:, :, 0], dftShift[:, :, 1])  # 幅度谱，中心化 (600, 600)
    sAmpLog = np.log10(1 + np.abs(sAmp))  # 幅度谱对数变换 (600, 600)
    # 傅里叶频谱沿半径的分布函数
    sRad = np.zeros((rmax,))  # (299,)
    sRad[0] = sAmp[x0, y0]
    for r in range(1, rmax):
        xc, yc = halfcircle(r, x0, y0)  # 半径为 r 的圆的整数坐标 (360,)
        sRad[r] = sum(sAmp[xc[i], yc[i]] for i in range(xc.shape[0]))  # (360,)
    sRadLog = np.log10(1 + np.abs(sRad))  # 极坐标幅度谱对数变换
    # 傅里叶频谱沿角度的分布函数
    xmax, ymax = halfcircle(rmax, x0, y0)  # 半径为 rmax 的圆的整数坐标 (360,)
    sAng = np.zeros((xmax.shape[0],))  # (360,)
    for a in range(xmax.shape[0]):  # xmax.shape[0]=(360,)
        xr, yr = intline(x0, xmax[a], y0, ymax[a])  # 从(x0,y0)到(xa,ya)线段所有点的坐标 (300,)
        sAng[a] = sum(sAmp[xr[i], yr[i]] for i in range(xr.shape[0]))  # (360,)
    return sAmpLog, sRadLog, sAng

if __name__ == '__main__':
    # 生成无序图像和有序图像
    gray1 = np.zeros((600, 600), np.uint8)
    gray2 = np.zeros((600, 600), np.uint8)
    num = 25
    pts = np.random.random([num, 2]) * 600  # 中心位置
    ang = np.random.random(num) * 180  # 旋转角度
    box = np.zeros((4, 2), np.int32)  # 计算旋转矩形的顶点, (4, 2)
    for i in range(num):
        # 有序方块，平行排列
        xc, yc = 100 * (i//5+1), 100 * (i%5+1)
        rect = ((xc, yc), (20, 40), 0)  # 旋转矩形类 ((cx,cy), (w,h), ang)
        box = np.int32(cv.boxPoints(rect))  # 旋转矩形的顶点, (4, 2)
        cv.drawContours(gray1, [box], 0, 255, 5)  # 将旋转矩形视为轮廓绘制
        # 无序方块，中心旋转
        rect = ((pts[i,0], pts[i,1]), (20, 40), ang[i])  # 位置与角度随机
        box = np.int32(cv.boxPoints(rect))
        cv.drawContours(gray2, [box], 0, 255, 5)

    # 图像纹理的频谱分析
    sAmpLog1, sRadLog1, sAng1 = specxture(gray1)
    sAmpLog2, sRadLog2, sAng2 = specxture(gray2)
    print(sAmpLog1.shape, sRadLog1.shape, sAng1.shape)

    plt.figure(figsize=(9, 5))
    plt.subplot(241), plt.imshow(gray1, 'gray')
    plt.axis('off'), plt.title("(1) Arranged blocks")
    plt.subplot(242), plt.imshow(sAmpLog1, 'gray')
    plt.axis('off'), plt.title("(2) Amp spectrum 1")
    plt.subplot(243), plt.xlim(0, 300), plt.yticks([])
    plt.plot(sRadLog1), plt.title("(3) S1 (radius)")
    plt.subplot(244), plt.xlim(0, 180), plt.yticks([])
    plt.plot(sAng1), plt.title("(4) S1 (theta)")
    plt.subplot(245), plt.imshow(gray2, 'gray')
    plt.axis('off'), plt.title("(5) Random blocks")
    plt.subplot(246), plt.imshow(sAmpLog2, 'gray')
    plt.axis('off'), plt.title("(6) Amp spectrum 2")
    plt.subplot(247), plt.xlim(0, 300), plt.yticks([])
    plt.plot(sRadLog2), plt.title("(7) S2 (radius)")
    plt.subplot(248), plt.xlim(0, 180), plt.yticks([])
    plt.plot(sAng2), plt.title("(8) S2 (theta)")
    plt.tight_layout()
    plt.show()

