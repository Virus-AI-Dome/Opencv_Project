"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【1706】特征检测之 ORB 算法
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv.imread("../images/Fig1701.png", flags=1)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    print("shape of image: ", gray.shape)

    # ORB 关键点检测
    orb = cv.ORB_create()  # 实例化 ORB 类
    # kp, descriptors = orb.detectAndCompute(gray)  # 检测关键点和生成描述符
    kp = orb.detect(img, None)  # 关键点检测，kp 为元组
    kp, des = orb.compute(img, kp)  # 生成描述符
    print("Num of keypoints: ", len(kp))  # 500
    print("Shape of kp descriptors: ", des.shape)  # (500,32)

    imgS = cv.convertScaleAbs(img, alpha=0.5, beta=128)
    imgKp1 = cv.drawKeypoints(imgS, kp, None)  # 只绘制关键点位置
    imgKp2 = cv.drawKeypoints(imgS, kp, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)  # 绘制关键点大小和方向
    plt.figure(figsize=(9, 3.5))
    plt.subplot(131), plt.title("(1) Original")
    plt.axis('off'), plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.subplot(132), plt.title("(2) ORB keypoints")
    plt.axis('off'), plt.imshow(cv.cvtColor(imgKp1, cv.COLOR_BGR2RGB))
    plt.subplot(133), plt.title("(3) ORB keypoints scaled")
    plt.axis('off'), plt.imshow(cv.cvtColor(imgKp2, cv.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()

