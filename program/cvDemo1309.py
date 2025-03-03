"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【1309】运动模糊退化图像的约束最小二乘滤波
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
    imgBlurred = np.fft.ifftshift(ifftBlurred)  # 逆中心化
    # imgBlurred = np.uint8(cv.normalize(np.abs(invShift), None, 0, 255, cv.NORM_MINMAX))
    return np.abs(imgBlurred)

def WienerFilterMotion(image, PSF, eps, K=0.01):  # 维纳滤波，K=0.01
    fftImg = np.fft.fft2(image.astype(np.float32))  # 傅里叶变换
    # fftPSF = np.fft.fft2(PSF) + eps  # 已知噪声功率
    fftPSF = np.fft.fft2(PSF)  # 未知噪声功率
    WienerFilter = np.conj(fftPSF) / (np.abs(fftPSF)**2 + K)  # Wiener 滤波器的传递函数
    fftImgFiltered = fftImg * WienerFilter  # Fuv = Huv * Filter
    ifftWienerFiltered = np.fft.ifft2(fftImgFiltered)  # 傅里叶逆变换
    WienerFiltered = np.fft.fftshift(ifftWienerFiltered)  # 逆中心化
    return np.abs(WienerFiltered)

def getPuv(image):  # 生成 P(u,v)
    h, w = image.shape[:2]
    hPad, wPad = h-3, w-3
    pxy = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])  # Laplacian kernel
    pxyPad = np.pad(pxy, ((hPad//2, hPad-hPad//2), (wPad//2, wPad-wPad//2)), mode='constant')
    fftPuv = np.fft.fft2(pxyPad)
    return fftPuv

def CLSFilterMotion(image, PSF, eps, gamma=0.01):  # 约束最小二乘滤波
    fftImg = np.fft.fft2(image.astype(np.float32))  # 傅里叶变换
    # fftPSF = np.fft.fft2(PSF) + eps  # 已知噪声功率
    fftPSF = np.fft.fft2(PSF)  # 未知噪声功率
    conj = fftPSF.conj()
    fftPuv = getPuv(image)
    CLSFilter = conj / (np.abs(fftPSF)**2 + gamma * (np.abs(fftPuv)**2))
    fftImgFiltered = fftImg * CLSFilter  # Fuv = Huv * Filter
    ifftCLSFiltered = np.fft.ifft2(fftImgFiltered)  # 傅里叶逆变换
    CLSFiltered = np.fft.fftshift(ifftCLSFiltered)  # 逆中心化
    return np.abs(CLSFiltered)


if __name__ == '__main__':
    # 读取原始图像
    img = cv.imread("../images/Fig1304.png", flags=0)
    hImg, wImg = img.shape[:2]
    # 生成不含噪声的运动模糊图像
    PSF = motionDSF(img, 30, 60)  # 运动模糊函数
    imgBlurred = getBlurredImg(img, PSF, 1e-6)  # 不含噪声的运动模糊图像
    # 生成带有噪声的运动模糊图像
    mu, std1 = 0.0, 0.1  # 高斯噪声的均值和标准差
    noiseGauss = np.random.normal(loc=mu, scale=std1, size=img.shape)  # 高斯噪声
    imgBlurNoisy1 = np.add(imgBlurred, noiseGauss)  # 添加高斯噪声
    mu, std2 = 0.0, 1.0  # 高斯噪声的均值和标准差
    noiseGauss = np.random.normal(loc=mu, scale=std2, size=img.shape)  # 高斯噪声
    imgBlurNoisy2 = np.add(imgBlurred, noiseGauss)  # 添加高斯噪声

    # 运动模糊退化图像的维纳滤波
    imgWienerF1 = WienerFilterMotion(imgBlurNoisy1, PSF, std1)
    imgWienerF2 = WienerFilterMotion(imgBlurNoisy2, PSF, std2)

    # 运动模糊退化图像的约束最小二乘滤波
    imgCLSFilter1 = CLSFilterMotion(imgBlurNoisy1, PSF, std1)
    imgCLSFilter2 = CLSFilterMotion(imgBlurNoisy2, PSF, std2)

    plt.figure(figsize=(9, 6))
    plt.subplot(231), plt.title("(1) Motion blurred (std=0.1)")
    plt.axis('off'), plt.imshow(imgBlurNoisy1, 'gray')
    plt.subplot(232), plt.title("(2) Wiener filter (std=0.1)")
    plt.axis('off'), plt.imshow(imgWienerF1, 'gray')
    plt.subplot(233), plt.title("(3) CLS filter (std=0.1)")
    plt.axis('off'), plt.imshow(imgCLSFilter1, 'gray')
    plt.subplot(234), plt.title("(4) Motion blurred (std=1.0)")
    plt.axis('off'), plt.imshow(imgBlurNoisy2, 'gray')
    plt.subplot(235), plt.title("(5) Wiener filter (std=1.0)")
    plt.axis('off'), plt.imshow(imgWienerF2, 'gray')
    plt.subplot(236), plt.title("(6) CLS filter (std=1.0)")
    plt.axis('off'), plt.imshow(imgCLSFilter2, 'gray')
    plt.tight_layout()
    plt.show()
