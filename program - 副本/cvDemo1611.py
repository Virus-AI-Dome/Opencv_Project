"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【1611】特征描述之 FREAK 描述符
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    # 读取图像
    img = cv.imread("../images/Fig1701.png", flags=1)  # 基准图像
    height, width = img.shape[:2]  # (500, 500)
    print("shape of image: ({},{})".format(height, width))

    # BRISK 检测关键点
    brisk = cv.BRISK_create()  # 创建 BRISK 检测器
    kp = brisk.detect(img)  # 关键点检测，kp 为元组
    print("Num of keypoints: ", len(kp))  # 271

    # BRIEF 特征描述
    brief = cv.xfeatures2d.BriefDescriptorExtractor_create()  # 实例化 BRIEF 类
    kpBrief, desBrief = brief.compute(img, kp)  # 计算 BRIEF 描述符
    print("BRIEF descriptors: ", desBrief.shape)  # (270, 32)

    # FREAK 特征描述
    freak = cv.xfeatures2d.FREAK_create()  # 实例化 FREAK 类
    kpFreak, desFreak = freak.compute(img, kp)  # 生成描述符
    print("FREAK descriptors: ", desFreak.shape)  # (196, 64)

    imgS = cv.convertScaleAbs(img, alpha=0.5, beta=128)
    imgKp1 = cv.drawKeypoints(imgS, kpBrief, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    imgKp2 = cv.drawKeypoints(imgS, kpFreak, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.figure(figsize=(9, 3.5))
    plt.subplot(131), plt.title("(1) Original")
    plt.axis('off'), plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.subplot(132), plt.title("(2) BRISK/BRIEF keypoints scaled")
    plt.axis('off'), plt.imshow(cv.cvtColor(imgKp1, cv.COLOR_BGR2RGB))
    plt.subplot(133), plt.title("(3) BRISK/FREAK keypoints scaled")
    plt.axis('off'), plt.imshow(cv.cvtColor(imgKp2, cv.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()
