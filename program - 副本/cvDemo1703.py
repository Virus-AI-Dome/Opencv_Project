"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【1703】特征检测之尺度不变特征变换（SIFT算法）
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv.imread("../images/Fig1701.png", flags=1)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # (512, 512)

    # # 比较：Harris 检测角点算法
    # dst = cv.cornerHarris(gray, 2, 3, k=0.04)
    # corners = np.column_stack(np.where(dst > 0.1 * dst.max()))  # 筛选并返回角点坐标 (y,x)
    # corners = corners.astype(np.int)  # (128, 2), 检测到的角点的点集 (y,x)
    # print("corners by Harris: ", corners.shape)
    # imgHarris = np.copy(img)
    # for point in corners:  # 注意坐标次序
    #     cv.circle(imgHarris, (point[1], point[0]), 4, (0, 0, 255), 1)  # 在点 (x,y) 处画圆

    # SIFT 关键点检测和特征描述
    # sift = cv.xfeatures2d.SIFT_create()  # OpenCV 早期版本
    sift = cv.SIFT.create()  # 实例化 SIFT 类
    # kp, descriptors = sift.detectAndCompute(gray)  # 检测关键点和生成描述符
    kp = sift.detect(gray)  # 关键点检测，kp 为元组
    kp, descriptors = sift.compute(gray, kp)  # 生成描述符
    print("Type of keypoints: {}\nType of descriptors: {}".format(type(kp), type(descriptors)))
    print("Num of keypoints: ", len(kp))  # 1184
    print("Coordinates of kp[0]: ", kp[0].pt)
    print("Keypoint diameter of kp[0]: ", kp[0].size)
    print("Keypoint orientation of kp[0]: ", kp[0].angle)
    print("Keypoint detector response on kp[0]: ", kp[0].response)
    print("Pyramid octave detected of kp[0]: ", kp[0].octave)
    print("Object id of kp[0]: ", kp[0].class_id)

    imgScale = cv.convertScaleAbs(img, alpha=0.5, beta=128)
    imgKp1 = cv.drawKeypoints(imgScale, kp, None, color=(0,0,0))  # 只绘制关键点位置
    imgKp2 = cv.drawKeypoints(imgScale, kp, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)  # 绘制关键点大小和方向
    plt.figure(figsize=(9, 3.4))
    plt.subplot(131), plt.title("(1) Original")
    plt.axis('off'), plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.subplot(132), plt.title("(2) SIFT keypoints")
    plt.axis('off'), plt.imshow(cv.cvtColor(imgKp1, cv.COLOR_BGR2RGB))
    plt.subplot(133), plt.title("(3) SIFT keypoints scaled")
    plt.axis('off'), plt.imshow(cv.cvtColor(imgKp2, cv.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()

