"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""


# 【0505】最低有效位数字盲水印
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv.imread("../images/Fig0301.png", flags=0)  # 灰度图像

    # 加载或生成水印信息
    # watermark = cv.imread("../images/logoCV.png", 0)  # 加载水印图片
    # markResize = cv.resize(watermark, img.shape[:2])  # 调整图片尺寸
    # _, binary = cv.threshold(markResize, 175, 1, cv.THRESH_BINARY)  # 二值图像
    binary = np.ones(img.shape[:2], np.uint8)
    cv.putText(binary, str(np.datetime64('today')), (50, 200), cv.FONT_HERSHEY_SIMPLEX, 2, 0, 2)
    cv.putText(binary, str(np.datetime64('now')), (50, 250), cv.FONT_HERSHEY_DUPLEX, 1, 0)
    cv.putText(binary, "Copyright: youcans@qq.com", (50, 300), cv.FONT_HERSHEY_DUPLEX, 1, 0)

    # 向原始图像嵌入水印
    # img: (p7,p6,...pg7,g6,...1,0) AND 254(11111110) -> imgH7: (p7,p6,...pg7,g6,...1,0)
    imgH7 = cv.bitwise_and(img, 1)  # 按位与运算，图像最低位 LSB=0
    # imgH7: (p7,p6,...pg7,g6,...1,0) OR b -> imgMark: (p7,p6,...pg7,g6,...1,b)
    imgMark = cv.bitwise_or(imgH7, binary)  # (p7,p6,...pg7,g6,...1,b)

    # 从嵌入水印图像中提取水印
    # extract = np.mod(imgMark, 2)  # 模运算，取图像的最低位 LSB
    extract = cv.bitwise_and(imgMark, 1 ) # 按位与运算，取图像的最低位 LSB

    plt.figure(figsize=(9, 3.5))
    plt.subplot(141), plt.title("1. Original"), plt.axis('off')
    plt.imshow(img, cmap='gray')
    plt.subplot(142), plt.title("3. Extracted imgH7"), plt.axis('off')
    plt.imshow(imgH7, cmap='gray')
    plt.subplot(143), plt.title("2. Embedded watermark"), plt.axis('off')
    plt.imshow(imgMark, cmap='gray')
    plt.subplot(144), plt.title("3. Extracted watermark"), plt.axis('off')
    plt.imshow(extract, cmap='gray')
    plt.tight_layout()
    plt.show()


