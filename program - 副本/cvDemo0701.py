"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【0701】图像的反转变换
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    filepath = "../images/Lena.tif"  # 读取文件的路径
    img = cv.imread(filepath, flags=1)  # 读取彩色图像(BGR)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 灰度变换

    # LUT 快速查表
    transTable = np.array([(255 - i) for i in range(256)]).astype("uint8")  # (256,)
    imgInv = cv.LUT(img, transTable)  # 彩色图像的反转变换
    grayInv = cv.LUT(gray, transTable)  # 灰度图像的反转变换
    print(img.shape, imgInv.shape, grayInv.shape)

    plt.figure(figsize=(9, 3.5))
    plt.subplot(131), plt.title("(1) Original"), plt.axis('off')
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.subplot(132), plt.title("(2) Invert image"), plt.axis('off')
    plt.imshow(cv.cvtColor(imgInv, cv.COLOR_BGR2RGB))
    plt.subplot(133), plt.title("(3) Invert gray"), plt.axis('off')
    plt.imshow(grayInv, cmap='gray')
    plt.tight_layout()
    plt.show()
