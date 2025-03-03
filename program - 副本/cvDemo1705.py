"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【1705】特征检测之 FAST 算法
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv.imread("../images/Fig1701.png", flags=1)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    print("shape of image: ", gray.shape)

    # FAST 关键点检测
    fast1 = cv.FastFeatureDetector_create()  # FAST 算法默认 t=10，使用非极大抑制
    # 默认值：threshold=10, nonmaxSuppression=true, type=FastFeatureDetector::TYPE_9_16
    kp1 = fast1.detect(gray)  # 特征点检测 t=10
    print("t={}, keypoints num={}".format(10, len(kp1)))
    # 设置比较阈值 t=20
    fastT2 = cv.FastFeatureDetector_create(threshold=20)  # 实例化 FAST 对象
    kp2 = fastT2.detect(gray)  # 特征点检测 t=20
    print("t={}, keypoints num={}".format(20, len(kp2)))
    # 设置比较阈值 t=30
    fastT3 = cv.FastFeatureDetector_create(threshold=30)  # 实例化 FAST 对象
    kp3 = fastT3.detect(gray)  # 特征点检测 t=30
    print("t={}, keypoints num={}".format(30, len(kp3)))

    imgS = cv.convertScaleAbs(img, alpha=0.5, beta=128)
    imgFAST1 = cv.drawKeypoints(imgS, kp1, None, color=(255, 0, 0))
    imgFAST2 = cv.drawKeypoints(imgS, kp2, None, color=(255, 0, 0))
    imgFAST3 = cv.drawKeypoints(imgS, kp3, None, color=(255, 0, 0))
    plt.figure(figsize=(9, 3.5))
    plt.subplot(131), plt.title("(1) FAST keypoints (t=10)")
    plt.axis('off'), plt.imshow(cv.cvtColor(imgFAST1, cv.COLOR_BGR2RGB))
    plt.subplot(132), plt.title("(2) FAST keypoints (t=20)")
    plt.axis('off'), plt.imshow(cv.cvtColor(imgFAST2, cv.COLOR_BGR2RGB))
    plt.subplot(133), plt.title("(3) FAST keypoints (t=30)")
    plt.axis('off'), plt.imshow(cv.cvtColor(imgFAST3, cv.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()

