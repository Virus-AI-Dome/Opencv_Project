"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""


# 【0505】最低有效位数字盲水印
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':

    img = cv.imread("../images/Lena.tif", flags=0)  # 灰度图像
    binary = np.ones(img.shape, np.uint8)
    cv.putText(binary, str(np.datetime64('today')), (50,200), cv.FONT_HERSHEY_SIMPLEX, 2, 0,  2)
    cv.putText(binary,str(np.datetime64('now')),(50,250), cv.FONT_HERSHEY_SIMPLEX, 1, 0)
    cv.putText(binary, "Copyright:@qq.com", (50,300), cv.FONT_HERSHEY_SIMPLEX, 1, 0)


    imgH7 = cv.bitwise_and(img,254)

    imgMark = cv.bitwise_or(imgH7, binary)

    extract = cv.bitwise_and(imgMark,1)

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





