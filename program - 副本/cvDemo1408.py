"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【1408】轮廓的形状特征
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv.imread("../images/Fig1403.png", flags=1)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 灰度图像
    print("shape of image:", gray.shape)

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

    # 按轮廓的面积排序，绘制面积最大的轮廓
    cnts = sorted(contours, key=cv.contourArea, reverse=True)  # 所有轮廓按面积排序
    cnt = cnts[0]  # 面积最大的轮廓
    imgCnt1 = img.copy()
    for i in range(len(cnts)):  # 注意 cnts 与 contours 的顺序不同
        if hierarchy[0,i,3] == -1:  # 最外层轮廓
            print("cnt[{}]: {}, area={}".format(i, cnts[i].shape, cv.contourArea(cnts[i])))

    # 轮廓的垂直矩形边界框
    imgCnt1 = img.copy()
    boundingBoxes = [cv.boundingRect(cnt) for cnt in contours]  # 所有轮廓的外接垂直矩形
    rect = cv.boundingRect(cnts[2])  # # ret 是元组，(x,y,w,h)
    x, y, w, h = rect  # 矩形左上顶点的坐标 (x,y), 矩形宽度 w, 高度 h
    print("Vertical rectangle: (x,y)={}, (w,h)={}".format((x, y), (w, h)))
    cv.rectangle(imgCnt1, (x,y), (x+w,y+h), (0,0,255), 3)  # 绘制垂直矩形边界框

    # 轮廓的最小矩形边界框
    rotRect = cv.minAreaRect(cnts[2])  # 返回值是元组，((x,y), (w,h), ang)
    boxPoints = np.int32(cv.boxPoints(rotRect))  # box 是2D点坐标向量的数组，(4, 2)
    cv.drawContours(imgCnt1, [boxPoints], 0, (0,255,0), 5)  # 将旋转矩形视为一个轮廓进行绘制
    # 矩形中心点 (x,y), 矩形宽度高度 (w,h), 旋转角度 ang，浮点数
    (x1, y1), (w1, h1), ang = np.int32(rotRect[0]), np.int32(rotRect[1]), int(rotRect[2])
    print("Minimum area rectangle: (Cx1,Cy1)={}, (w,h)={}, ang={})".format((x1,y1), (w1,h1), ang))

    # 轮廓的最小外接圆
    center, r = cv.minEnclosingCircle(cnts[1])  # center 是元组 (cx,cy), 半径 r
    Cx, Cy, radius = int(center[0]), int(center[1]), int(r)
    cv.circle(imgCnt1, (Cx, Cy), radius, (0, 255, 0), 2)
    print("Minimum circle: (Cx,Cy)=({},{}), r={}".format(Cx, Cy, radius))

    # 轮廓的最小外接三角形
    points = np.float32(cnts[0])  # 输入 points 必须为32位浮点数
    areaTri, triangle = cv.minEnclosingTriangle(points)  # area 三角形面积, triangle 三角形顶点 (3,1,2)
    print("Area of minimum enclosing triangle: {:.1f}".format(areaTri))
    intTri = np.int32(triangle)  # triangle 三角形顶点 (3,1,2)
    cv.polylines(imgCnt1, [intTri], True, (255, 0, 0), 2)

    # 轮廓近似多边形
    imgCnt2 = img.copy()
    epsilon = 0.01 * cv.arcLength(cnts[1], True)  # 以轮廓周长的 1% 作为近似距离
    approx = cv.approxPolyDP(cnts[1], epsilon, True)  # approx (15, 1, 2)
    cv.polylines(imgCnt2, [approx], True, (0, 0, 255), 3)  # 绘制近似多边形

    # 轮廓的拟合椭圆
    ellipRect = cv.fitEllipse(cnts[2])  # 返回值是元组，((x,y), (w,h), ang)
    boxPoints = np.int32(cv.boxPoints(ellipRect))  # boxPoints 是2D点坐标向量的数组，(4, 2)
    cv.drawContours(imgCnt2, [boxPoints], 0, (0,255,255), 2)  # # 将旋转矩形视为一个轮廓进行绘制
    cv.ellipse(imgCnt2, ellipRect, (0,255,255), 3)
    (x2,y2), (w2,h2), ang = np.int32(ellipRect[0]), np.int32(ellipRect[1]), int(ellipRect[2])
    print("Fitted ellipse: (Cx2,Cy2)={}, (w,h)={}, ang={})".format((x2,y2), (w2,h2), ang))
    # 对比：近似椭圆外接的旋转矩形，不是轮廓的最小外接旋转矩形
    rotRect = cv.minAreaRect(cnts[2])  # 最小外接旋转矩形
    boxPoints = np.int32(cv.boxPoints(rotRect))
    cv.drawContours(imgCnt2, [boxPoints], 0, (0,255,0), 2)

    # # 拟合直线
    rows, cols = img.shape[:2]
    [vx, vy, x, y] = cv.fitLine(cnts[0], cv.DIST_L1, 0, 0.01, 0.01)
    lefty = int((-x * vy/vx) + y)
    righty = int(((cols - x) * vy/vx) + y)
    cv.line(imgCnt2, (0,lefty), (cols-1,righty), (255,0,0), 3)

    # 检查轮廓是否为凸面体
    isConvex = cv.isContourConvex(cnts[0])  # True 凸形, False 非凸
    print("cnts[1] is ContourConvex?", isConvex)
    # 获取轮廓的凸壳
    hull1 = cv.convexHull(cnts[0], returnPoints=True)  # 返回凸壳顶点坐标
    cv.polylines(imgCnt2, [hull1], True, (0, 0, 255), 3)  # 绘制多边形
    print("hull.shape: ", hull1.shape)  # 凸壳顶点坐标 (x,y), (24, 1, 2)
    hull2 = cv.convexHull(cnts[0], returnPoints=False)  # 返回凸壳顶点在cnt的索引序号
    print("hull.shape: ", hull2.shape)  # 凸壳顶点坐标 (x,y), (24, 1)

    plt.figure(figsize=(9, 3.5))
    plt.subplot(131), plt.axis('off'), plt.title("(1) Binary image")
    plt.imshow(binary, 'gray')
    plt.subplot(132), plt.axis('off'), plt.title("(2) Enclosing geometry")
    plt.imshow(cv.cvtColor(imgCnt1, cv.COLOR_BGR2RGB))
    plt.subplot(133), plt.axis('off'), plt.title("(3) Approximate geometry")
    plt.imshow(cv.cvtColor(imgCnt2, cv.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()
