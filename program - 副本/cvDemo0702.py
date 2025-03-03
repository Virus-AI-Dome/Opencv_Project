"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【0702】图像的线性灰度变换
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    gray = cv.imread("../images/Lena.tif", flags=0)  # 读取为灰度图像
    h, w = gray.shape[:2]  # 图片的高度和宽度

    # 线性变换参数 dst = a*src + b
    a1, b1 = 1, 50  # a=1,b>0: 灰度值上移
    a2, b2 = 1, -50  # a=1,b<0: 灰度值下移
    a3, b3 = 1.5, 0  # a>1,b=0: 对比度增强
    a4, b4 = 0.8, 0  # 0<a<1,b=0: 对比度减小
    a5, b5 = -0.5, 0  # a<0,b=0: 暗区域变亮，亮区域变暗
    a6, b6 = -1, 255  # a=-1,b=255: 灰度值反转

    # 灰度线性变换
    timeBegin = cv.getTickCount()
    img1 = cv.convertScaleAbs(gray, alpha=a1, beta=b1)
    img2 = cv.convertScaleAbs(gray, alpha=a2, beta=b2)
    img3 = cv.convertScaleAbs(gray, alpha=a3, beta=b3)
    img4 = cv.convertScaleAbs(gray, alpha=a4, beta=b4)
    img5 = cv.convertScaleAbs(gray, alpha=a5, beta=b5)
    img6 = cv.convertScaleAbs(gray, alpha=a6, beta=b6)
    # img1 = cv.add(gray, b1)
    # img2 = cv.add(gray, b2)
    # img3 = cv.multiply(gray, a3)
    # img4 = cv.multiply(gray, a4)
    # img5 = np.abs(a5*gray)
    # img6 = np.clip((a6*gray+b6), 0, 255)  # 截断函数
    timeEnd = cv.getTickCount()
    time = (timeEnd - timeBegin) / cv.getTickFrequency()
    print("Grayscale transformation by OpenCV: {} sec".format(round(time, 4)))

    # 二重循环遍历
    timeBegin = cv.getTickCount()
    for i in range(h):
        for j in range(w):
            img1[i][j] = min(255, max((gray[i][j] + b1), 0))  # a=1,b>0: 颜色发白
            img2[i][j] = min(255, max((gray[i][j] + b2), 0))  # a=1,b<0: 颜色发黑
            img3[i][j] = min(255, max(a3 * gray[i][j], 0))  # a>1,b=0: 对比度增强
            img4[i][j] = min(255, max(a4 * gray[i][j], 0))  # 0<a<1,b=0: 对比度减小
            img5[i][j] = min(255, max(abs(a5 * gray[i][j] + b5), 0))  # a=-0.5,b=0
            img6[i][j] = min(255, max(abs(a6 * gray[i][j] + b6), 0))  # a=-1,b=255
    timeEnd = cv.getTickCount()
    time = (timeEnd - timeBegin) / cv.getTickFrequency()
    print("Grayscale transformation by nested loop: {} sec".format(round(time, 4)))

    plt.figure(figsize=(9, 6))
    titleList = ["a=1, b=50", "a=1, b=-50", "a=1.5, b=0",
                 "a=0.8, b=0", "a=-0.5, b=0", "a=-1, b=255"]
    imageList = [img1, img2, img3, img4, img5, img6]
    for k in range(len(imageList)):
        plt.subplot(2, 3, k+1), plt.title("({}) {}".format(k+1, titleList[k]))
        plt.axis('off'), plt.imshow(imageList[k], vmin=0, vmax=255, cmap='gray')
    plt.tight_layout()
    plt.show()
