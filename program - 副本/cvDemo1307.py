"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【1307】湍流模糊退化图像的逆滤波
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def turbulenceBlur(img, k=0.001):
    # 湍流模糊传递函数: H(u,v) = exp(-k(u^2+v^2)^5/6)
    M, N = img.shape[1], img.shape[0]  # width, height
    u, v = np.meshgrid(np.arange(M), np.arange(N))
    radius = (u-M//2)**2 + (v-N//2)**2
    kernel = np.exp(-k * np.power(radius, 5/6))
    return kernel

def getDegradedImg(image, Huv, D0=0):  # 根据退化模型生成退化图像
    # (1) 傅里叶变换, 中心化
    fft = np.fft.fft2(image.astype(np.float32))  # 傅里叶变换
    fftShift = np.fft.fftshift(fft)  # 中心化
    # (2) 在频率域修改傅里叶变换
    fftShiftFilter = fftShift * Huv  # Guv = Fuv * Huv
    # (3) 逆中心化，傅里叶逆变换
    invShift = np.fft.ifftshift(fftShiftFilter)  # 逆中心化
    imgIfft = np.fft.ifft2(invShift)  # 逆傅里叶变换，返回值是复数数组
    imgDegraded = np.uint8(cv.normalize(np.abs(imgIfft), None, 0, 255, cv.NORM_MINMAX))
    return imgDegraded

def ideaLPFilter(img, radius=10):  # 理想低通滤波器
    M, N = img.shape[1], img.shape[0]  # width, height
    u, v = np.meshgrid(np.arange(M), np.arange(N))
    D = np.sqrt((u-M//2)**2 + (v-N//2)**2)
    kernel = np.zeros(img.shape[:2], np.float32)
    kernel[D<=radius] = 1.0
    return kernel

def invFilterTurb(image, Huv, D0=0):  # 基于模型的逆滤波
    # (1) 傅里叶变换, 中心化
    fftImg = np.fft.fft2(image.astype(np.float32))  # 傅里叶变换
    fftShift = np.fft.fftshift(fftImg)  # 中心化
    # (2) 在频率域修改傅里叶变换
    if D0==0:
        fftShiftFilter = fftShift / Huv  # Guv = Fuv / Huv
    else:  # 理想低通滤波器在 D0 截止
        lpFilter = ideaLPFilter(image, radius=D0)
        fftShiftFilter = fftShift / Huv * lpFilter  # Guv = Fuv / Huv
    # (3) 逆中心化，傅里叶逆变换
    invShift = np.fft.ifftshift(fftShiftFilter)  # 逆中心化
    imgIfft = np.fft.ifft2(invShift)  # 逆傅里叶变换，返回值是复数数组
    imgRestored = np.uint8(cv.normalize(np.abs(imgIfft), None, 0, 255, cv.NORM_MINMAX))
    return imgRestored

if __name__ == '__main__':
    # 读取原始图像
    img = cv.imread("../images/Fig1303.png", 0)

    # 生成湍流模糊图像
    Hturb = turbulenceBlur(img, k=0.003)  # 湍流传递函数
    imgTurb = np.abs(getDegradedImg(img, Hturb, 0.0))
    hImg, wImg = img.shape[:2]
    print(hImg, wImg)
    print(imgTurb.max(), imgTurb.min())

    # 逆滤波
    imgRestored = invFilterTurb (imgTurb, Hturb, hImg)  # Huv
    imgRestored1 = invFilterTurb (imgTurb, Hturb, D0=40)  # 在 D0 之外截止
    imgRestored2 = invFilterTurb (imgTurb, Hturb, D0=60)
    imgRestored3 = invFilterTurb (imgTurb, Hturb, D0=80)

    plt.figure(figsize=(9, 6))
    plt.subplot(231), plt.title("(1) Original")
    plt.axis('off'), plt.imshow(img, 'gray')
    plt.subplot(232), plt.title("(2) Turbulence blur")
    plt.axis('off'), plt.imshow(imgTurb, 'gray')
    plt.subplot(233), plt.title("(3) Inverse filter (D0=full)")
    plt.axis('off'), plt.imshow(imgRestored, 'gray')
    plt.subplot(234), plt.title("(4) Inverse filter (D0=40)")
    plt.axis('off'), plt.imshow(imgRestored1, 'gray')
    plt.subplot(235), plt.title("(5) Inverse filter (D0=60)")
    plt.axis('off'), plt.imshow(imgRestored2, 'gray')
    plt.subplot(236), plt.title("(6) Inverse filter (D0=80)")
    plt.axis('off'), plt.imshow(imgRestored3, 'gray')
    plt.tight_layout()
    plt.show()
