"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

#【1302】霍夫变换直线检测
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv.imread("../images/Fig1201.png", flags=1)  # 彩色图像
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 灰度图像
    hImg, wImg = gray.shape

    # (1) Canny 边缘检测，TL, TH 为低阈值、高阈值
    TL, ratio = 60, 3  # ratio=TH/TL
    imgGauss = cv.GaussianBlur(gray, (5, 5), 0)
    imgCanny = cv.Canny(imgGauss, TL, TL*ratio)

    # (3) 标准霍夫直线检测
    imgEdge1 = cv.convertScaleAbs(img, alpha=0.25, beta=192)
    lines = cv.HoughLines(imgCanny, 1, np.pi/180, threshold=100)  # (n, 1, 2)
    print("cv.HoughLines: ", lines.shape)  # 每行元素 (i,1,:) 表示直线参数 rho, theta
    for i in range(lines.shape[0]//2):  # 绘制出部分检测直线
        rho, theta = lines[i, 0, :]  # lines 每行 2 个元素
        if (theta<(np.pi/4)) or (theta>(3*np.pi/4)):  # 直线与图像上下相交
            pt1 = (int(rho/np.cos(theta)), 0)  # (x,0), 直线与顶侧的交点
            pt2 = (int((rho - hImg*np.sin(theta))/np.cos(theta)), hImg)  # (x,h), 直线与底侧的交点
            cv.line(imgEdge1, pt1, pt2, (255,127,0), 2)  # 绘制直线
        else:  # 直线与图像左右相交
            pt1 = (0, int(rho/np.sin(theta)))  # (0,y), 直线与左侧的交点
            pt2 = (wImg, int((rho - wImg*np.cos(theta))/np.sin(theta)))  # (w,y), 直线与右侧的交点
            cv.line(imgEdge1, pt1, pt2, (127,0,255), 2)  # 绘制直线
        # print("rho={}, theta={:.1f}".format(rho, theta))

    # (4) 累积概率霍夫变换
    imgEdge2 = cv.convertScaleAbs(img, alpha=0.25, beta=192)
    minLineLength = 30  # 检测直线的最小长度
    maxLineGap = 10  # 直线上像素的最大间隔
    lines = cv.HoughLinesP(imgCanny, 1, np.pi/180, 60, minLineLength, maxLineGap)  # lines: (n,1,4)
    print("cv.HoughLinesP: ", lines.shape)  # 每行元素 (i,1,:) 表示参数 x1, y1, x2, y2
    for line in lines:
        x1, y1, x2, y2 = line[0]  # 返回值每行是一个4元组，表示直线端点 (x1, y1, x2, y2)
        cv.line(imgEdge2, (x1,y1), (x2,y2), (255,0,0), 2)  # 绘制直线
        # print("(x1,y1)=({},{}), (x2,y2)=({},{})".format(x1, y1, x2, y2))

    plt.figure(figsize=(9, 3.3))
    plt.subplot(131), plt.axis('off'), plt.title("(1) Canny edges")
    plt.imshow(cv.bitwise_not(imgCanny), cmap='gray')
    plt.subplot(132), plt.axis('off'), plt.title("(2) cv.HoughLines")
    plt.imshow(cv.cvtColor(imgEdge1, cv.COLOR_RGB2BGR))
    plt.subplot(133), plt.axis('off'), plt.title("(3) cv.HoughLinesP")
    plt.imshow(cv.cvtColor(imgEdge2, cv.COLOR_RGB2BGR))
    plt.tight_layout()
    plt.show()
