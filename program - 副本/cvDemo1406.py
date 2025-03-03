"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【1406】查找和绘制图像的轮廓
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv.imread("../images/Fig1402.png", flags=1)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 灰度图像
    _, binary = cv.threshold(gray, 127, 255, cv.THRESH_OTSU + cv.THRESH_BINARY_INV)

    # 寻找二值化图中的轮廓
    # binary, contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)  # OpenCV3
    contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)  # OpenCV4~
    # print("len(contours): ", len(contours))  # contours 是列表，只有长度没有形状
    print("hierarchy.shape: ", hierarchy.shape)  # 层次结构

    # # 绘制全部轮廓
    contourTree = img.copy()  # OpenCV 某些版本会修改原始图像
    contourTree = cv.drawContours(contourTree, contours, -1, (255,255,255), 5)  # OpenCV3

    #  绘制最外层轮廓和最内层轮廓
    imgContour = img.copy()
    for i in range(len(contours)):  # 绘制第 i 个轮廓
        x, y, w, h = cv.boundingRect(contours[i])  # 外接矩形
        text = "{}#({},{})".format(i, x, y)
        contourTree = cv.putText(contourTree, text, (x, y), cv.FONT_HERSHEY_DUPLEX, 0.8, (0,0,0))
        print("i={}\tcontours[{}]:{}\thierarchy[0,{}]={}"
              .format(i, i, contours[i].shape, i, hierarchy[0][i]))
        # text = "{}#".format(i)
        # if i<3:
        #     contourTree = cv.putText(contourTree, text, (x+10, y-10), cv.FONT_HERSHEY_DUPLEX, 0.8, (0,0,0))
        # elif i==3:
        #     contourTree = cv.putText(contourTree, text, (x+10, y+30), cv.FONT_HERSHEY_DUPLEX, 0.8, (0,0,0))
        # else:
        #     contourTree = cv.putText(contourTree, text, (x+10, y-10), cv.FONT_HERSHEY_DUPLEX, 0.8, (0,0,0))
        print("i={}\tcontours[{}]:{}\thierarchy[0,{}]={}"
              .format(i, i, contours[i].shape, i, hierarchy[0][i]))

        if hierarchy[0,i,2]==-1:  # 最内层轮廓
            imgContour = cv.drawContours(imgContour, contours, i, (0,0,255), thickness=-1)  # 内部填充
        if hierarchy[0,i,3]==-1:  # 最外层轮廓
            imgContour = cv.drawContours(imgContour, contours, i, (255,255,255), thickness=5)

    plt.figure(figsize=(9, 3.2))
    plt.subplot(131), plt.axis('off'), plt.title("(1) Original")
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.subplot(132), plt.axis('off'), plt.title("(2) Contours")
    plt.imshow(cv.cvtColor(contourTree, cv.COLOR_BGR2RGB))
    plt.subplot(133), plt.axis('off'), plt.title("(3) Selected contour")
    plt.imshow(cv.cvtColor(imgContour, cv.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()

