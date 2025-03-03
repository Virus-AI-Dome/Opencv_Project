"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【1407】轮廓的基本参数：面积、周长、质心、等效直径、极端点
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv.imread("../images/Fig1403.png", flags=1)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 灰度图像

    # HSV 色彩空间图像分割
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)  # 将图片转换到 HSV 色彩空间
    lowerBlue, upperBlue = np.array([100, 43, 46]), np.array([124, 255, 255])  # 蓝色阈值
    segment = cv.inRange(hsv, lowerBlue, upperBlue)  # 背景色彩图像分割
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))  # (5, 5) 结构元
    binary = cv.dilate(cv.bitwise_not(segment), kernel=kernel, iterations=3)  # 图像膨胀

    # 对二值图像查找轮廓
    # binary, contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)  # OpenCV3
    contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)  # OpenCV4~
    print("len(contours) = ", len(contours))  # 所有轮廓的列表

    # 绘制全部轮廓，contourIdx=-1 绘制全部轮廓
    imgCnts = img.copy()
    for i in range(len(contours)):  # 绘制第 i 个轮廓
        if hierarchy[0,i,3] == -1:  # 最外层轮廓
            moments = cv.moments(contours[i])  # Mu：几何矩, 中心矩和归一化矩
            cx = int(moments['m10'] / moments['m00'])  # 轮廓的质心 (Cx,Cy)
            cy = int(moments['m01'] / moments['m00'])
            text = "{}:({},{})".format(i, cx, cy)
            cv.drawContours(imgCnts, contours, i, (205,205,205), -1)  # 绘制轮廓, 内部填充
            cv.circle(imgCnts, (cx, cy), 5, (0,0,255), -1)  # 在轮廓的质心上绘制圆点
            cv.putText(imgCnts, text, (cx, cy), cv.FONT_HERSHEY_DUPLEX, 0.8, (0,0,255))
            print("contours[{}]:{}\ttext={}".format(i, contours[i].shape, text))

    # 按轮廓的面积排序，绘制面积最大的轮廓
    cnts = sorted(contours, key=cv.contourArea, reverse=True)  # 所有轮廓按面积排序
    for i in range(len(cnts)):  # 注意 cnts 与 contours 的顺序不同
        if hierarchy[0,i,3] == -1:  # 最外层轮廓
            print("cnt[{}]: {}, area={}".format(i, cnts[i].shape, cv.contourArea(cnts[i])))

    # 轮廓的面积 (area)
    cnt = cnts[0]  # 面积最大的轮廓
    imgCntMax = img.copy()
    cv.drawContours(imgCntMax, cnts, 0, (0,0,255), 5)  # 绘制面积最大的轮廓
    area = cv.contourArea(cnt)  # 轮廓的面积
    moments = cv.moments(cnt)  # 图像的矩
    print("Area of contour: ", area)
    print("Area by moments['m00']: ", moments['m00'])

    # 轮廓的等效直径 (Equivalent diameter)
    dEqu = round(np.sqrt(4*area/np.pi), 2)  # 轮廓的等效直径
    print("Equivalent diameter:", dEqu)

    # 轮廓的质心 (centroid)：(Cx,Cy)
    if moments['m00'] > 0:
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        print("Centroid of contour: ({}, {})".format(cx, cy))
        cv.circle(imgCntMax, (cx, cy), 8, (0,0,255), -1)  # 在轮廓的质心上绘制圆点
    else:
        print("Error: moments['m00']=0 .")

    # 轮廓的周长 (Perimeter)
    perimeter = cv.arcLength(cnt, True)  # True  表示输入是闭合轮廓
    print("Perimeter of contour: {:.1f}".format(perimeter))

    # 轮廓的极端点位置
    leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])  # cnt[:,:,0], 所有边界点的横坐标
    rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
    topmost = tuple(cnt[cnt[:,:,1].argmin()][0])  # cnt[:,:,1], 所有边界点的纵坐标
    bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
    print("Left most is {} at Pos{}".format(leftmost[0], leftmost))
    print("Right most is {} at Pos{}".format(rightmost[0], rightmost))
    print("Top most is {} at Pos{}".format(topmost[1], topmost))
    print("Bottom most is {} at Pos{}".format(bottommost[1], bottommost))
    for point in [leftmost, rightmost, topmost, bottommost]:
        cv.circle(imgCntMax, point, 8, (0,255,0), -1)  # 在轮廓的极端点上绘制圆点

    plt.figure(figsize=(9, 3.5))
    plt.subplot(131), plt.axis('off'), plt.title("(1) Original")
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.subplot(132), plt.axis('off'), plt.title("(2) Contours")
    plt.imshow(cv.cvtColor(imgCnts, cv.COLOR_BGR2RGB))
    plt.subplot(133), plt.axis('off'), plt.title("(3) Maximum contour")
    plt.imshow(cv.cvtColor(imgCntMax, cv.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()
