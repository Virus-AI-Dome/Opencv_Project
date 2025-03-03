"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【0907】绿屏抠图和更换背景颜色
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    # 在HSV空间对绿屏色彩区域进行阈值处理，生成遮罩进行抠图
    img = cv.imread("../images/Fig0903.png", flags=1)  # 读取彩色图像
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)  # 转换到 HSV 色彩空间

    # 在 HSV 空间检查指定颜色的区域范围，生成二值遮罩
    lowerColor = (35, 43, 46)  # (下限: 绿色33/43/46)
    upperColor = (77, 255, 255)  # (上限: 绿色77/255/255)
    binary = cv.inRange(hsv, lowerColor, upperColor)  # 指定颜色为白色
    # 绿屏抠图
    mask = cv.bitwise_not(binary)  # 掩模图像，绿屏背景为白色，前景为黑色
    matting = cv.bitwise_or(img, img, mask=mask)  # 生成抠图图像 (前景保留，背景黑色)
    # 基于抠图掩模更换背景颜色
    imgReplace = matting.copy()
    imgReplace[mask==0] = [0, 0, 255]  # 黑色背景区域(0/0/0) 修改为红色 (BGR:0/0/255)

    plt.figure(figsize=(9, 5.8))
    plt.subplot(221), plt.title("(1) Original"), plt.axis('off')
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.subplot(222), plt.title("(2) Background"), plt.axis('off')
    plt.imshow(binary, cmap='gray')
    plt.subplot(223), plt.title("(3) Matting"), plt.axis('off')
    plt.imshow(cv.cvtColor(matting, cv.COLOR_BGR2RGB))
    plt.subplot(224), plt.title("(4) Red background"), plt.axis('off')
    plt.imshow(cv.cvtColor(imgReplace, cv.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()


