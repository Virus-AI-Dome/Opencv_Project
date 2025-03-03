"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【0806】限制对比度的自适应局部直方图均衡化
import cv2 as cv
from matplotlib import pyplot as plt

if __name__ == '__main__':
    gray = cv.imread("../images/Fig0803.png", flags=0)  # 读取为灰度图像
    imgEequa = cv.equalizeHist(gray)  # 全局直方图均衡化

    # 限制对比度的自适应局部直方图均衡化
    Clahe = cv.createCLAHE(clipLimit=100, tileGridSize=(16, 16))  # 创建 CLAHE 对象
    imgClahe = Clahe.apply(gray)  # 应用 CLANE 方法

    plt.figure(figsize=(9, 3.5))
    plt.subplot(131), plt.axis('off')
    plt.title("(1) Original")
    plt.imshow(gray, cmap='gray', vmin=0, vmax=255)
    plt.subplot(132), plt.axis('off')
    plt.title("(2) Global Equalize Hist")
    plt.imshow(imgEequa, cmap='gray', vmin=0, vmax=255)
    plt.subplot(133), plt.axis('off')
    plt.title("(3) Local Equalize Hist")
    plt.imshow(imgClahe, cmap='gray', vmin=0, vmax=255)
    plt.tight_layout()
    plt.show()

    # plt.figure(figsize=(9, 5.5))
    # plt.subplot(231), plt.title('Original'), plt.axis('off')
    # plt.imshow(gray, cmap='gray', vmin=0, vmax=255)
    # plt.subplot(232), plt.title('Global Equalize Hist'), plt.axis('off')
    # plt.imshow(imgEequa, cmap='gray', vmin=0, vmax=255)
    # plt.subplot(233), plt.title('Local Equalize Hist'), plt.axis('off')
    # plt.imshow(imgClahe, cmap='gray', vmin=0, vmax=255)
    # plt.subplot(234, xticks=[], yticks=[])
    # histGray = cv.calcHist([gray], [0], None, [256], [0, 256])
    # plt.plot(histGray), plt.xlim([0, 256])
    # plt.subplot(235, xticks=[], yticks=[])
    # histEequa = cv.calcHist([imgEequa], [0], None, [256], [0, 256])
    # plt.plot(histEequa), plt.xlim([0, 256])
    # plt.subplot(236, xticks=[], yticks=[])
    # histClahe = cv.calcHist([imgClahe], [0], None, [256], [0, 256])
    # plt.plot(histClahe), plt.xlim([0, 256])
    # plt.tight_layout()
    # plt.show()



