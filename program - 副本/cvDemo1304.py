"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【1304】离散雷登变换的正弦图
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

if __name__ == '__main__':
    # 生成原始图像
    img1 = np.zeros((512,512), np.uint8)
    cv.rectangle(img1, (50,100), (100,150), 255, -1)
    cv.rectangle(img1, (50,200), (100,300), 255, -1)
    cv.rectangle(img1, (50,350), (100,400), 255, -1)
    cv.rectangle(img1, (226,150), (336,352), 255, -1)

    img2 = np.zeros((512,512), np.uint8)
    cv.circle(img2, (200, 150), 30, 255, -1)  # 绘制圆
    cv.circle(img2, (312, 150), 30, 255, -1)  # 绘制圆
    cv.ellipse(img2, (256,320), (100,50), 0, 0, 360, 255, 30)  # 绘制椭圆

    img3 = cv.imread("../images/Fig1302.png", flags=0)  # 灰度图像

    # 离散雷登变换
    imgRadon1 = disRadonTransform(img1, img1.shape[0])
    imgRadon2 = disRadonTransform(img2, img2.shape[0])
    imgRadon3 = disRadonTransform(img3, img3.shape[0])
    print(img1.shape, imgRadon1.shape)

    plt.figure(figsize=(9, 6))
    plt.subplot(231), plt.axis('off')  # 绘制原始图像
    plt.title("(1) Demo image 1"), plt.imshow(img1, 'gray')
    plt.subplot(232), plt.axis('off')  # 绘制原始图像
    plt.title("(2) Demo image 2"), plt.imshow(img2, 'gray')
    plt.subplot(233), plt.axis('off')  # 绘制原始图像
    plt.title("(3) Demo image 3"), plt.imshow(img3, 'gray')
    plt.subplot(234), plt.axis('off')  # 绘制 sinogram 图
    plt.title("(4) Radon transform 1"), plt.imshow(imgRadon1, 'gray')
    plt.subplot(235), plt.axis('off')  # 绘制 sinogram 图
    plt.title("(5) Radon transform 2"), plt.imshow(imgRadon2, 'gray')
    plt.subplot(236), plt.axis('off')  # 绘制 sinogram 图
    plt.title("(6) Radon transform 3"), plt.imshow(imgRadon3, 'gray')
    plt.tight_layout()
    plt.show()
