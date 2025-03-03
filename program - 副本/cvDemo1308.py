"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【1308】运动模糊退化图像的维纳滤波
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def motionDSF(image, angle, dist):  # 运动模糊退化模型
    hImg, wImg = image.shape[:2]
    xCenter = (wImg-1)/2
    yCenter = (hImg-1)/2
    sinVal = np.sin(angle * np.pi / 180)
    cosVal = np.cos(angle * np.pi / 180)
    PSF = np.zeros((hImg, wImg))  # 点扩散函数
    for i in range(dist):  # 将对应角度上motion_dis个点置成1
        xOffset = round(sinVal * i)
        yOffset = round(cosVal * i)
        PSF[int(xCenter-xOffset), int(yCenter+yOffset)] = 1
    return PSF / PSF.sum()  # 归一化

def getBlurredImg(image, PSF, eps=1e-6):  # 对图片进行运动模糊
    fftImg = np.fft.fft2(image.astype(np.float32))  # 傅里叶变换
    # fftShift = np.fft.fftshift(fft)  # 中心化
    fftPSF = np.fft.fft2(PSF) + eps
    fftFilter = fftImg * fftPSF  # Guv = Fuv * Huv
    ifftBlurred = np.fft.ifft2(fftFilter)  # 傅里叶逆变换
    blurred = np.fft.ifftshift(ifftBlurred)  # 逆中心化
    imgBlurred = np.abs(blurred)
    return imgBlurred

def invFilterMotion(image, PSF, eps):  # 运动模糊的逆滤波
    fftImg = np.fft.fft2(image.astype(np.float32))  # 傅里叶变换
    # fftShift = np.fft.fftshift(fft)  # 中心化
    fftPSF = np.fft.fft2(PSF) + eps  # 已知噪声功率
    fftInvFiltered = fftImg / fftPSF  # Fuv = Huv / Guv
    ifftInvFiltered = np.fft.ifft2(fftInvFiltered)  # 傅里叶逆变换
    invFiltered = np.fft.fftshift(ifftInvFiltered)  # 逆中心化
    return np.abs(invFiltered)

def WienerFilterMotion(image, PSF, eps, K=0.01):  # 维纳滤波，K=0.01
    fftImg = np.fft.fft2(image.astype(np.float32))  # 傅里叶变换
    # fftPSF = np.fft.fft2(PSF) + eps  # 已知噪声功率
    fftPSF = np.fft.fft2(PSF)  # 未知噪声功率
    WienerFilter = np.conj(fftPSF) / (np.abs(fftPSF)**2 + K)  # Wiener 滤波器的传递函数
    fftImgFiltered = fftImg * WienerFilter  # Fuv = Huv * Filter
    ifftWienerFiltered = np.fft.ifft2(fftImgFiltered)  # 傅里叶逆变换
    WienerFiltered = np.fft.fftshift(ifftWienerFiltered)  # 逆中心化
    return np.abs(WienerFiltered)

if __name__ == '__main__':
    # 读取原始图像
    img = cv.imread("../images/Fig1304.png", flags=0)
    hImg, wImg = img.shape[:2]

    # (1) 不含噪声运动模糊退化图像的复原
    # 生成不含噪声的运动模糊图像
    PSF = motionDSF(img, 30, 60)  # 运动模糊函数
    imgBlurred = getBlurredImg(img, PSF, 1e-6)  # 不含噪声的运动模糊图像
    # 退化图像复原
    imgInvF = invFilterMotion(imgBlurred, PSF, 1e-6)  # 对运动模糊图像逆滤波
    imgWienerF = WienerFilterMotion(imgBlurred, PSF, 1e-6)  # 对运动模糊图像维纳滤波

    # (2) 加性噪声运动模糊退化图像的复原
    # 生成带有噪声的运动模糊图像
    mu, scale = 0.0, 0.5  # 高斯噪声的均值和标准差
    noiseGauss = np.random.normal(loc=mu, scale=scale, size=img.shape)  # 高斯噪声
    imgBlurNoisy = np.add(imgBlurred, noiseGauss)  # 添加高斯噪声
    # 退化图像复原
    imgInvFNoisy = invFilterMotion(imgBlurNoisy, PSF, scale)  # 对噪声模糊图像逆滤波
    imgWienerFNoisy = WienerFilterMotion(imgBlurNoisy, PSF, scale)  # 对噪声模糊图像逆滤波维纳滤波

    plt.figure(figsize=(9, 6))
    plt.subplot(231), plt.title("(1) Motion blurred")
    plt.axis('off'), plt.imshow(imgBlurred, 'gray')
    plt.subplot(232), plt.title("(2) Inverse filter")
    plt.axis('off'), plt.imshow(imgInvF, 'gray')
    plt.subplot(233), plt.title("(3) Wiener filter")
    plt.axis('off'), plt.imshow(imgWienerF, 'gray')
    plt.subplot(234), plt.title("(4) Noisy motion blurred")
    plt.axis('off'), plt.imshow(imgBlurNoisy, 'gray')
    plt.subplot(235), plt.title("(5) Noisy inverse filter")
    plt.axis('off'), plt.imshow(imgInvFNoisy, 'gray')
    plt.subplot(236), plt.title("(6) Noisy Wiener filter")
    plt.axis('off'), plt.imshow(imgWienerFNoisy, 'gray')
    plt.tight_layout()
    plt.show()
