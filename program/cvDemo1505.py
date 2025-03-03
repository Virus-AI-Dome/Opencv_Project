"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【1505】基于轮廓标记的分水岭算法
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv.imread("../images/Fig0301.png", flags=1)  # 彩色图像(BGR)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 转为灰度图像

    # (1) 梯度处理
    imgGauss = cv.GaussianBlur(gray, (5, 5), -1)
    grad = cv.Canny(imgGauss, 50, 150)  # Canny 梯度算子
    # gx = cv.Sobel(imgGauss, cv.CV_32F, 1, 0, ksize=3)  # SobelX 水平梯度
    # gy = cv.Sobel(imgGauss, cv.CV_32F, 0, 1, ksize=3)  # SobelY 垂直梯度
    # mag, angle = cv.cartToPolar(gx, gy, angleInDegrees=1)  # 梯度幅值和角度
    # ret, grad = cv.threshold(np.uint8(mag), 127, 255, cv.THRESH_OTSU)

    # (2) 查找和绘制图像轮廓
    contours, hierarchy = cv.findContours(grad, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # 查找图像轮廓
    markers = np.zeros(img.shape[:2], np.int32)  # 生成标识图像，所有轮廓区域标识为索引号 (index)
    for index in range(len(contours)):  # 用轮廓的索引号 index 标识轮廓区域
        markers = cv.drawContours(markers, contours, index, (index, index, index), 1, 8, hierarchy)
    ContoursMarkers = np.zeros(img.shape[:2], np.uint8)
    ContoursMarkers[markers>0] = 255  # 轮廓图像，将所有轮廓区域标识为白色 (255)
    print(len(contours))

    # (3) 分水岭算法
    markers = cv.watershed(img, markers)  # 分水岭算法，所有轮廓的像素点被标注为 -1
    WatershedMarkers = cv.convertScaleAbs(markers)

    # (4) 用随机颜色填充分割图像
    bgrMarkers = np.zeros_like(img)
    for i in range(len(contours)):  # 用随机颜色进行填充
        colorKind = np.random.randint(0, 255, size=(1, 3))
        bgrMarkers[markers == i] = colorKind
    bgrFilled = cv.addWeighted(img, 0.67, bgrMarkers, 0.33, 0)  # 填充后与原始图像融合

    plt.figure(figsize=(9, 6))
    plt.subplot(231), plt.axis('off'), plt.title("(1) Original")
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.subplot(232), plt.axis('off'), plt.title("(2) Gradient")
    plt.imshow(grad, 'gray')  # Canny 梯度算子
    plt.subplot(233), plt.axis('off'), plt.title("(3) Contours markers")
    plt.imshow(ContoursMarkers, 'gray')  # 轮廓
    plt.subplot(234), plt.axis('off'), plt.title("(4) Watershed markers")
    plt.imshow(WatershedMarkers, 'gray')  # 确定背景
    plt.subplot(235), plt.axis('off'), plt.title("(5) Colorful Markers")
    plt.imshow(cv.cvtColor(bgrMarkers, cv.COLOR_BGR2RGB))
    plt.subplot(236), plt.axis('off'), plt.title("(6) Cutted image")
    plt.imshow(cv.cvtColor(bgrFilled, cv.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()
