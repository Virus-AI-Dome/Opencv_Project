"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【0901】图像阈值处理之固定阈值法
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    # 生成灰度图像
    hImg, wImg = 512, 512
    img = np.zeros((hImg, wImg), np.uint8)  # # 创建黑色图像
    cv.rectangle(img, (60, 60), (450, 320), (127, 127, 127), -1)  # 矩形填充
    cv.circle(img, (256, 256), 120, (205, 205, 205), -1)  # 圆形填充
    # 添加高斯噪声
    mu, sigma = 0.0, 20.0
    noiseGause = np.random.normal(mu, sigma, img.shape)
    imgNoise = np.add(img, noiseGause)
    imgNoise = np.uint8(cv.normalize(imgNoise, None, 0, 255, cv.NORM_MINMAX))

    # 阈值处理
    _, imgBin1 = cv.threshold(imgNoise, 63, 255, cv.THRESH_BINARY)  # thresh=63
    _, imgBin2 = cv.threshold(imgNoise, 125, 255, cv.THRESH_BINARY)  # thresh=125
    _, imgBin3 = cv.threshold(imgNoise, 175, 255, cv.THRESH_BINARY)  # thresh=175

    plt.figure(figsize=(9, 6))
    plt.subplot(231), plt.axis('off'), plt.title("(1) Original"), plt.imshow(img, 'gray')
    plt.subplot(232), plt.axis('off'), plt.title("(2) Noisy image"), plt.imshow(imgNoise, 'gray')
    histCV = cv.calcHist([imgNoise], [0], None, [256], [0, 256])
    plt.subplot(233, yticks=[]), plt.title("(3) Gray hist")
    plt.bar(range(256), histCV[:, 0]), plt.axis([0, 255, 0, np.max(histCV)])
    plt.subplot(234), plt.axis('off'), plt.title("(4) threshold=63"), plt.imshow(imgBin1, 'gray')
    plt.subplot(235), plt.axis('off'), plt.title("(5) threshold=125"), plt.imshow(imgBin2, 'gray')
    plt.subplot(236), plt.axis('off'), plt.title("(6) threshold=175"), plt.imshow(imgBin3, 'gray')
    plt.tight_layout()
    plt.show()

