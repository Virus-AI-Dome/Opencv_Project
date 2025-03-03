"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【1305】雷登变换反投影重建图像
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

    plt.figure(figsize=(9, 6))
    plt.subplot(231), plt.title("(1) Demo image 1")
    plt.axis('off'), plt.imshow(img1, 'gray')  # 绘制原始图像
    plt.subplot(232), plt.title("(2) Radon transform 1")
    plt.axis('off'), plt.imshow(imgRadon1, 'gray')  # 绘制 sinogram 图
    plt.subplot(233), plt.title("(3) Inverse RadonTrans 1")
    plt.axis('off'), plt.imshow(imgInvRadon1, 'gray')
    plt.subplot(234), plt.title("(4) Demo image 1")
    plt.axis('off'), plt.imshow(img2, 'gray')
    plt.subplot(235), plt.title("(5) Radon transform 2")
    plt.axis('off'), plt.imshow(imgRadon2, 'gray')
    plt.subplot(236), plt.title("(6) Inverse RadonTrans 2")
    plt.axis('off'), plt.imshow(imgInvRadon2, 'gray')
    plt.tight_layout()
    plt.show()

