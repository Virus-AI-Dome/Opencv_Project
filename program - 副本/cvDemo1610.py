"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【1610】特征描述之 BRIEF 关键点描述符
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    # 读取图像
    # img = cv.imread("../images/Fig1601.png", flags=1)  # 基准图像
    img = cv.imread("../images/Fig1701.png", flags=1)  # 基准图像
    height, width = img.shape[:2]  # (540, 600)
    print("shape of image: ({},{})".format(height, width))

    # STAR 关键点检测
    star = cv.xfeatures2d.StarDetector_create()  # STAR 特征检测
    kpStar = star.detect(img, None)  # STAR 特征检测
    print("Num of keypoints: ", len(kpStar))

    # BRIEF 特征描述
    brief1 = cv.xfeatures2d.BriefDescriptorExtractor_create(bytes=16)  # 实例化BRIEF类
    kpBriefStar, des = brief1.compute(img, kpStar)  # 计算 BRIEF 描述符
    print("Shape of kp descriptors (bytes=16): ", des.shape)

    brief2 = cv.xfeatures2d.BriefDescriptorExtractor_create()  # 实例化BRIEF类，默认 bytes=32
    kpBriefStar, des = brief2.compute(img, kpStar)  # 通过 BRIEF 计算描述子
    print("Shape of kp descriptors (bytes=32): ", des.shape)

    imgS = cv.convertScaleAbs(img, alpha=0.5, beta=128)
    imgKp1 = cv.drawKeypoints(imgS, kpBriefStar, None, color=(0,0,0))
    imgKp2 = cv.drawKeypoints(imgS, kpBriefStar, None, color=(0,0,0), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.figure(figsize=(9, 3.5))
    plt.subplot(131), plt.title("(1) Original")
    plt.axis('off'), plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.subplot(132), plt.title("(2) Star/BRIEF keypoints")
    plt.axis('off'), plt.imshow(cv.cvtColor(imgKp1, cv.COLOR_BGR2RGB))
    plt.subplot(133), plt.title("(3) Star/BRIEF keypoints scaled")
    plt.axis('off'), plt.imshow(cv.cvtColor(imgKp2, cv.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()