"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

#【1402】边缘检测之 LoG 算子 (Marr-Hildreth 算法)
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def ZeroDetect(img):  # 判断零交叉点
    h, w = img.shape[0], img.shape[1]
    zeroCrossing = np.zeros_like(img, np.uint8)
    for x in range(0, w - 1):
        for y in range(0, h - 1):
            if img[y][x] < 0:
                if (img[y][x - 1] > 0) or (img[y][x + 1] > 0) \
                        or (img[y - 1][x] > 0) or (img[y + 1][x] > 0):
                    zeroCrossing[y][x] = 255
    return zeroCrossing

if __name__ == '__main__':
    img = cv.imread("../images/Fig1401.png", flags=0)  # 灰度图像
    imgBlur = cv.boxFilter(img, -1, (3, 3))  # Blur 平滑去噪

    # (1) 近似的 Marr-Hildreth 卷积核 (5*5)
    kernel_MH5 = np.array([
        [0, 0, -1, 0, 0],
        [0, -1, -2, -1, 0],
        [-1, -2, 16, -2, -1],
        [0, -1, -2, -1, 0],
        [0, 0, -1, 0, 0]])
    from scipy import signal
    imgMH5 = signal.convolve2d(imgBlur, kernel_MH5, boundary='symm', mode='same')  # 卷积计算
    # kFlipMH5 = cv.flip(kernel_MH5, -1)  # # 翻转卷积核
    # imgMH5 = cv.filter2D(img, -1, kFlipMH5)  # 注意不能使用 cv.filter2D 实现
    zeroMH5 = ZeroDetect(imgMH5)  # 判断零交叉点

    # (2) 由 Gauss 标准差计算 Marr-Hildreth 卷积核
    sigma = 3  # Gauss 标准差，模糊尺度
    size = int(2 * round(3*sigma)) + 1  # 根据标准差确定窗口大小，3*sigma 占比 99.7%
    print("sigma={:d}, size={}".format(sigma, size))
    x, y = np.meshgrid(np.arange(-size/2+1, size/2+1), np.arange(-size/2+1, size/2+1))  # 生成网格
    norm2 = x*x + y*y
    sigma2, sigma4 = np.power(sigma, 2), np.power(sigma, 4)
    kernelLoG = ((norm2 - (2.0*sigma2))/sigma4) * np.exp(-norm2/(2.0*sigma2))  # 计算 LoG 卷积核
    imgLoG = signal.convolve2d(imgBlur, kernelLoG, boundary='symm', mode='same')  # 卷积计算
    zeroCross1 = ZeroDetect(imgLoG)  # 判断零交叉点

    # (3) 高斯滤波后使用拉普拉斯算子
    imgGauss = cv.GaussianBlur(imgBlur, (0,0), sigmaX=2)
    imgGaussLap = cv.Laplacian(imgGauss, cv.CV_32F, ksize=3)
    zeroCross2 = ZeroDetect(imgGaussLap)  # 判断零交叉点

    plt.figure(figsize=(9, 6))
    plt.subplot(231), plt.title("(1) LoG (sigma=0.5)")
    plt.axis('off'), plt.imshow(imgMH5, cmap='gray')
    plt.subplot(234), plt.title("(4) Zero crossing (size=5)")
    plt.axis('off'), plt.imshow(zeroMH5, cmap='gray')
    plt.subplot(232), plt.title("(2) LoG (sigma=2)")
    plt.axis('off'), plt.imshow(cv.convertScaleAbs(imgGaussLap), cmap='gray')
    plt.subplot(235), plt.title("(5) Zero crossing (size=13)")
    plt.axis('off'), plt.imshow(zeroCross2, cmap='gray')
    plt.subplot(233), plt.title("(3) LoG (sigma=3)")
    plt.axis('off'), plt.imshow(imgLoG, cmap='gray')
    plt.subplot(236), plt.title("(6) Zero crossing (size=19)")
    plt.axis('off'), plt.imshow(zeroCross1, cmap='gray')
    plt.tight_layout()
    plt.show()
