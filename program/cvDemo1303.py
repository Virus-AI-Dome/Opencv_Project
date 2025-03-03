"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

#【1303】霍夫变换圆检测
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv.imread("../images/Fig1204.png", flags=1)  # 彩色图像
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 灰度图像
    imgGauss = cv.GaussianBlur(gray, (3, 3), 0)

    # (1) Canny 边缘检测，TL, TH 为低阈值、高阈值
    TL, TH = 25, 50  # ratio=2
    imgCanny = cv.Canny(imgGauss, TL, TH)

    # (2) 霍夫变换圆检测
    circles = cv.HoughCircles(imgGauss, cv.HOUGH_GRADIENT, 1, 40, param1=50, param2=30, minRadius=20, maxRadius=80)
    circlesVal = np.uint(np.squeeze(circles))  # 删除数组维度，(1,12,3)->(12,3)
    print(circles.shape, circlesVal.shape)

    # (3) 将检测到的圆绘制在原始图像中
    imgFade = cv.convertScaleAbs(img, alpha=0.5, beta=128)
    for i in range(len(circlesVal)):
        x, y, r = circlesVal[i]
        print("i={}, x={}, y={}, r={}".format(i, x, y, r))
        cv.circle(imgFade, (x,y), r, (255, 0, 0), 2)  # 绘制圆
        cv.circle(imgFade, (x,y), 2, (0, 0, 255), 8)  # 圆心

    plt.figure(figsize=(9, 3.2))
    plt.subplot(131), plt.axis('off'), plt.title("(1) Original")
    plt.imshow(gray, cmap='gray')
    plt.subplot(132), plt.axis('off'), plt.title("(2) Canny edge2")
    plt.imshow(cv.bitwise_not(imgCanny), cmap='gray')
    plt.subplot(133), plt.axis('off'), plt.title("(3) Hough circles")
    plt.imshow(cv.cvtColor(imgFade, cv.COLOR_RGB2BGR))
    plt.tight_layout()
    plt.show()
