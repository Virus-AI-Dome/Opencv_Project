"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【1704】特征检测之加速鲁棒特征变换（SURF算法）
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv.imread("../images/Fig1701.png", flags=1)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # (500, 500)
    print("shape of image: ", gray.shape)

    # SURF 关键点检测和特征描述
    surf = cv.xfeatures2d.SURF_create()  # 实例化 SURF 对象
    # kp, descriptors = surf.detectAndCompute(gray)  # 检测关键点和生成描述符
    kpSurf = surf.detect(gray)  # 关键点检测
    kpSurf, desSurf = surf.compute(gray, kpSurf)  # 生成描述符
    print("Num of keypoints: ", len(kpSurf))  # 695

    # # 构造 BRISK 对象，检测关键点，计算特征描述向量
    # brisk = cv.BRISK_create()  # 创建 BRISK 检测器
    # kpSurf = brisk.detect(gray)  # 关键点检测，kp 为元组
    # kpSurf, descriptors = brisk.compute(gray, kpSurf)  # 生成描述符
    # print("Num of keypoints: ", len(kpSurf), descriptors.shape)

    imgScale = cv.convertScaleAbs(img, alpha=0.5, beta=128)
    # imgScale = img.copy()
    imgSurf1 = cv.drawKeypoints(imgScale, kpSurf, None, color=(0,0,0))  # 只绘制关键点位置
    imgSurf2 = cv.drawKeypoints(imgScale, kpSurf, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)  # 绘制关键点大小和方向

    plt.figure(figsize=(9, 3.4))
    plt.subplot(131), plt.title("(1) Original")
    plt.axis('off'), plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.subplot(132), plt.title("(2) SURF keypoints")
    plt.axis('off'), plt.imshow(cv.cvtColor(imgSurf1, cv.COLOR_BGR2RGB))
    plt.subplot(133), plt.title("(3) SURF keypoints scaled")
    plt.axis('off'), plt.imshow(cv.cvtColor(imgSurf2, cv.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()

