"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【1102】Numpy实现图像傅里叶变换
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    # (1) 创建原始图像
    img = cv.imread("../images/Fig1101.png", flags=0)  # 读取灰度图像

    # (2) np.fft.fft2 实现二维傅里叶变换
    # fft = np.fft.fft2(img.astype(np.float32))
    fft = np.fft.fft2(img)  # 傅里叶变换，fft 是复数数组 (512, 512)
    fftShift = np.fft.fftshift(fft)  # 中心化，将低频分量移到频谱中心

    # (3) np.fft.ifft2 实现二维傅里叶逆变换
    iFftShift = np.fft.ifftshift(fftShift)  # 逆中心化，将低频逆转换回四角
    ifft = np.fft.ifft2(iFftShift)  # 逆傅里叶变换，ifft 是复数数组 (512, 512)
    rebuild = np.abs(ifft)  # 重建图像，复数的模
    print("img：{}, fft:{}, ifft:{}".format(img.shape, fft.shape, ifft.shape))

    # (4) 傅里叶频谱的显示
    fftAmp = np.abs(fft)  # 复数的模，幅度谱
    ampLog = np.log(1 + fftAmp)  # 幅度谱对数变换
    shiftFftAmp = np.abs(fftShift)  # 中心化幅度谱
    shiftAmpLog = np.log(1 + shiftFftAmp)  # 中心化幅度谱对数变换
    # phase = np.arctan2(fft.imag, fft.real)  # 计算相位角(弧度)
    phase = np.angle(fft)  # 复数的幅角(弧度)
    fftPhi = phase / np.pi * 180  # 转换为角度制 [-180, 180]
    print("img min/max：{}, {}".format(img.min(), img.max()))
    print("fftAmp min/max：{:.1f}, {:.1f}".format(fftAmp.min(), fftAmp.max()))
    print("fftPhi min/max：{:.1f}, {:.1f}".format(fftPhi.min(), fftPhi.max()))
    print("ampLog min/max: {:.1f}, {:.1f}".format(ampLog.min(), ampLog.max()))
    print("rebuild min/max: {:.1f}, {:.1f}".format(rebuild.min(), rebuild.max()))

    plt.figure(figsize=(9, 6))
    plt.subplot(231), plt.title("(1) Original")
    plt.imshow(img, cmap='gray'), plt.axis('off')
    plt.subplot(232), plt.title("(2) FFT Phase"), plt.axis('off')
    plt.imshow(fftPhi, cmap='gray'), plt.axis('off')
    plt.subplot(233), plt.title("(3) FFT amplitude")
    plt.imshow(fftAmp, cmap='gray'), plt.axis('off')
    plt.subplot(234), plt.title("(4) LogTrans of amplitude")
    plt.imshow(ampLog, cmap='gray'), plt.axis('off')
    plt.subplot(235), plt.title("(5) Shift to center")
    plt.imshow(shiftAmpLog, cmap='gray'), plt.axis('off')
    plt.subplot(236), plt.title("(6) Rebuild image with IFFT")
    plt.imshow(rebuild, cmap='gray'), plt.axis('off')
    plt.tight_layout()
    plt.show()

