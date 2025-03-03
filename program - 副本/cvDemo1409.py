"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【1409】轮廓的属性
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv.imread("../images/Fig1402.png", flags=1)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 灰度图像
    _, binary = cv.threshold(gray, 127, 255, cv.THRESH_OTSU + cv.THRESH_BINARY_INV)

    # (1) 寻找二值图像中的轮廓
    # binary, contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)  # OpenCV3
    contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)  # OpenCV4~
    print("len(contours) = ", len(contours))  # 所有轮廓的列表
    cnts = sorted(contours, key=cv.contourArea, reverse=True)  # 所有轮廓按面积排序
    cnt = cnts[-1]  # 面积最小的轮廓

    # (2) 轮廓的宽高比（Aspect Ratio）
    xv, yv, wv, hv = cv.boundingRect(cnt)  # 轮廓的垂直矩形边界框
    aspectRatio = round(wv / hv, 2)  # 轮廓外接垂直矩形的宽高比
    print("Vertical rectangle: w={}, h={}".format(wv, hv))
    print("Aspect ratio:", aspectRatio)

    # (3) 轮廓的面积比（Extent）
    areaRect = wv * hv  # 垂直矩形边界框的面积，wv * hv
    areaCnt = cv.contourArea(cnt)  # 轮廓的面积
    extent = round(areaCnt / areaRect, 2)  # 轮廓的面积比
    print("Area of cnt:", areaCnt)
    print("Area of VertRect:", areaRect)
    print("Extent(area ratio):", extent)

    # (4) 轮廓的坚实度（Solidity）
    areaCnt = cv.contourArea(cnt)  # 轮廓的面积
    hull = cv.convexHull(cnt)  # 轮廓的凸包，返回凸包顶点集
    areaHull = cv.contourArea(hull)  # 凸包的面积
    solidity = round(areaCnt / areaHull, 2)  # 轮廓的坚实度
    print("Area of cnt:", areaCnt)
    print("Area of convex hull:", areaHull)
    print("Solidity(area ratio):", solidity)

    # (5) 轮廓的等效直径 (Equivalent diameter)
    areaCnt = cv.contourArea(cnt)  # 轮廓的面积
    dEqu = round(np.sqrt(areaCnt * 4 / np.pi), 2)  # 轮廓的等效直径
    print("Area of cnt:", areaCnt)
    print("Equivalent diameter:", dEqu)

    # (6) 轮廓的方向 (Orientation)
    elliRect = cv.fitEllipse(cnt)  # 旋转矩形类，elliRect[2] 是旋转角度 ang
    angle = round(elliRect[2], 1)  # 轮廓的方向，椭圆与水平轴的夹角
    print("Orientation of cnt: {} deg".format(angle))

    # (7) 轮廓的掩模和像素点 (Mask)
    maskCnt = np.zeros(gray.shape, np.uint8)  # 背景区域置为黑色
    cv.drawContours(maskCnt, [cnt], 0, 255, -1)  # 轮廓区域置为白色
    pixelsNP = np.transpose(np.nonzero(maskCnt))  # (15859, 2): (y, x)
    pixelsCV = cv.findNonZero(maskCnt)  # (15859, 1, 2): (x, y)
    print("pixelsNP: {}, pixelsCV: {}".format(pixelsNP.shape, pixelsCV.shape))

    # (8) 轮廓的最大值/最小值及其位置
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(gray, mask=maskCnt)  # 必须用灰度图像
    print("Minimum value is {} at Pos{}".format(min_val, min_loc))
    print("Maximum value is {} at Pos{}".format(max_val, max_loc))

    # (9) 轮廓的灰度均值和颜色均值
    meanGray = cv.mean(gray, maskCnt)  # (mg, 0, 0, 0)
    meanImg = cv.mean(img, maskCnt)  # (mR, mG, mB, 0)
    print("Gray mean of cnt: {:.1f}".format(meanGray[0]))
    print("BGR mean: ({:.1f}, {:.1f}, {:.1f})".format(meanImg[0], meanImg[1], meanImg[2]))
