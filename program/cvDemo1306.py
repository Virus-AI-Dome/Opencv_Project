"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【1306】雷登变换滤波反投影重建图像
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def disRadonTransform(image, steps):
    hImg, wImg = image.shape[:2]
    channels = len(image[0])
    center = (hImg//2, wImg//2)
    resCV = np.zeros((channels, channels), dtype=np.float32)
    for s in range(steps):
        # 以 center 为中心旋转，逆时针旋转角度 ang，黑色填充
        ang = -s*180/steps
        MatRot = cv.getRotationMatrix2D(center, ang, 1.0)
        rotationCV = cv.warpAffine(image, MatRot, (hImg, wImg), borderValue=0)
        resCV[:, s] = sum(rotationCV.astype(np.float32))
    transRadon = cv.normalize(resCV, None, 0, 255, cv.NORM_MINMAX)
    return transRadon

def invRadonTransform(image, steps):  # 雷登变换反投影
    hImg, wImg = image.shape[:2]
    channels = len(image[0])
    center = (hImg//2, wImg//2)
    # steps = min(hImg,wImg)
    res = np.zeros((steps, channels, channels))
    for s in range(steps):
        expandDims = np.expand_dims(image[:, s], axis=0)
        repeat = expandDims.repeat(channels, axis=0)
        ang = s*180/steps
        MatRot = cv.getRotationMatrix2D(center, ang, 1.0)
        res[s] = cv.warpAffine(repeat, MatRot, (hImg, wImg), borderValue=0)
    invTransRadon = np.sum(res, axis=0)
    return invTransRadon

def SLFilter(N, d):  # SL 滤波器, Sinc 函数对斜坡滤波器进行截断
    rangeN = np.arange(N)
    filterSL = - 2 / (np.pi**2 * d**2 * (4*(rangeN-N/2)**2 - 1))
    return filterSL

def filterInvRadonTransform(image, steps):  # 滤波反投影重建图像
    hImg, wImg = image.shape[:2]
    channels = len(image[0])
    center = (channels//2, channels//2)
    # steps = min(hImg,wImg)
    res = np.zeros((steps, channels, channels))
    filterSL = SLFilter(channels, 1)  # SL 滤波器
    for s in range(steps):
        sImg = image[:, s]  # 投影值
        sImgFiltered = np.convolve(filterSL, sImg, "same")  # 投影值和 SL 滤波器进行卷积
        filterExpandDims = np.expand_dims(sImgFiltered, axis=0)
        filterRepeat = filterExpandDims.repeat(hImg, axis=0)
        ang = s*180/steps
        MatRot = cv.getRotationMatrix2D(center, ang, 1.0)
        res[s] = cv.warpAffine(filterRepeat, MatRot, (hImg, wImg), borderValue=0)
    filterInvRadon = np.sum(res, axis=0)
    return filterInvRadon


if __name__ == '__main__':
    # 读取原始图像
    img1 = np.zeros((512,512), np.uint8)
    cv.circle(img1, (200, 150), 30, 255, -1)  # 绘制圆
    cv.circle(img1, (312, 150), 30, 255, -1)  # 绘制圆
    cv.ellipse(img1, (256,320), (100,50), 0, 0, 360, 255, 30)  # 绘制椭圆

    img2 = cv.imread("../images/Fig1302.png", flags=0)  # 灰度图像

    # 雷登变换
    imgRadon1 = disRadonTransform(img1, img1.shape[0])
    imgRadon2 = disRadonTransform(img2, img2.shape[0])

    # 雷登变换反投影
    imgInvRadon1 = invRadonTransform(imgRadon1, imgRadon1.shape[0])
    imgInvRadon2 = invRadonTransform(imgRadon2, imgRadon2.shape[0])

    # 滤波反投影重建图像
    imgFilterInvRadon1 = filterInvRadonTransform(imgRadon1, imgRadon1.shape[0])
    imgFilterInvRadon2 = filterInvRadonTransform(imgRadon2, imgRadon2.shape[0])

    plt.figure(figsize=(9, 6))
    plt.subplot(231), plt.title("(1) Demo image 1")
    plt.axis('off'), plt.imshow(img1, 'gray')
    plt.subplot(232), plt.title("(2) Inv-Radon Transform 1")
    plt.axis('off'), plt.imshow(imgInvRadon1, 'gray')
    plt.subplot(233), plt.title("(3) Filterd Inv-Radon 1")
    plt.axis('off'), plt.imshow(imgFilterInvRadon1, 'gray')
    plt.subplot(234), plt.title("(4) Demo image 2")
    plt.axis('off'), plt.imshow(img2, 'gray')
    plt.subplot(235), plt.title("(5) Inv-Radon Transform2")
    plt.axis('off'), plt.imshow(imgInvRadon2, 'gray')
    plt.subplot(236), plt.title("(6) Filterd Inv-Radon 2")
    plt.axis('off'), plt.imshow(imgFilterInvRadon2, 'gray')
    plt.tight_layout()
    plt.show()

