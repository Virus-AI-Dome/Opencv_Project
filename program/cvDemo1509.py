"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【1509】运动目标跟踪之帧间差分法
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def capMovement(img, imgMove, cntSize):
    # 检查图像尺寸是否一致
    imgCap = imgMove.copy()
    if (imgMove.shape != img.shape):
        print("Error in diffienent image size.")
    if img.shape[-1]==3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        imgMove = cv.cvtColor(imgMove, cv.COLOR_BGR2GRAY)

    # 与参考图像差分比较
    thresh = 20  # 比较阈值，大于阈值时判定存在差异
    absSub = cv.absdiff(img, imgMove)
    _, mask = cv.threshold(absSub, thresh, 255, cv.THRESH_BINARY)  # 阈值分割
    # 提取运动目标
    dilate = cv.dilate(mask, None, iterations=1)
    cnts, _ = cv.findContours(dilate, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # 查找轮廓
    print(len(cnts))
    for contour in cnts:
        if cv.contourArea(contour) > cntSize*5:  # 很大轮廓红色标识
            # print(cv.contourArea(contour))
            (x, y, w, h) = cv.boundingRect(contour)
            cv.rectangle(imgCap, (x, y), (x+w, y+h), (0, 0, 255), 5)  # 绘制矩形，红色
        elif cv.contourArea(contour) > cntSize:  # 忽略较小的轮廓区域
            (x, y, w, h) = cv.boundingRect(contour)
            # print(x, y, w, h)
            cv.rectangle(imgCap, (x, y), (x+w, y+h), (255, 0, 0), -1)  # 绘制矩形，蓝色
    return mask, imgCap

if __name__ == '__main__':
    # 创建视频读取/捕获对象
    img0 = cv.imread("../images/FVid1.png", flags=1)  # 静止参考图像
    img1 = cv.imread("../images/FVid2.png", flags=1)  # 读取相邻帧
    img2 = cv.imread("../images/FVid3.png", flags=1)
    img3 = cv.imread("../images/FVid4.png", flags=1)
    mask1, imgCap1 = capMovement(img0, img1, cntSize=1000)
    mask2, imgCap2 = capMovement(img1, img2, cntSize=1000)
    mask3, imgCap3 = capMovement(img2, img3, cntSize=1000)

    plt.figure(figsize=(9, 5.5))
    plt.subplot(231), plt.axis('off'), plt.title("(1) MoveCapture1")
    plt.imshow(cv.cvtColor(imgCap1, cv.COLOR_BGR2RGB))
    plt.subplot(232), plt.axis('off'), plt.title("(2) MoveCapture2")
    plt.imshow(cv.cvtColor(imgCap2, cv.COLOR_BGR2RGB))
    plt.subplot(233), plt.axis('off'), plt.title("(3) MoveCapture3")
    plt.imshow(cv.cvtColor(imgCap3, cv.COLOR_BGR2RGB))
    plt.subplot(234), plt.axis('off'), plt.title("(4) Diffabs mask1")
    plt.imshow(mask1, 'gray')
    plt.subplot(235), plt.axis('off'), plt.title("(5) Diffabs mask2")
    plt.imshow(mask2, 'gray')
    plt.subplot(236), plt.axis('off'), plt.title("(6) Diffabs mask3")
    plt.imshow(mask3, 'gray')
    plt.tight_layout()
    plt.show()
