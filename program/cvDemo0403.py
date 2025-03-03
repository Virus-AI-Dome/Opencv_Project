"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【0403】绘制倾斜的矩形
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    height, width, channels = 300, 400, 3
    img = np.ones((height, width, channels), np.uint8) * 192  # 创建黑色图像 RGB=0

    # (1) 围绕矩形中心旋转
    cx, cy, w, h = (200, 150, 200, 100)  # 左上角坐标 (x,y), 宽度 w，高度 h
    img1 = img.copy()
    cv.circle(img1, (cx, cy), 2, (0, 0, 255), 10)  # 旋转中心
    angle = [15, 30, 45, 60, 75, 90]  # 旋转角度，顺时针方向
    box = np.zeros((4, 2), np.int32)  # 计算旋转矩形的顶点, (4, 2)
    for i in range(len(angle)):
        rect = ((cx, cy), (w, h), angle[i])  # Box2D：中心点 (x,y), 矩形宽度高度 (w,h), 旋转角度 angle
        box = np.int32(cv.boxPoints(rect))  # 计算旋转矩形的顶点, (4, 2)
        color = (30 * i, 0, 255 - 30 * i)
        cv.drawContours(img1, [box], 0, color, 1)  # 将旋转矩形视为轮廓绘制
        print(rect)

    # (2) 围绕矩形左上顶点旋转
    x, y, w, h = (200, 100, 160, 100)  # 左上角坐标 (x,y), 宽度 w，高度 h
    img2 = img.copy()
    cv.circle(img2, (x, y), 4, (0, 0, 255), -1)  # 旋转中心
    angle = [15, 30, 45, 60, 75, 90, 120, 150, 180, 225]  # 旋转角度，顺时针方向
    for i in range(len(angle)):
        ang = angle[i] * np.pi / 180
        x1, y1 = x, y
        x2 = int(x + w * np.cos(ang))
        print("pain",w * np.cos(ang))
        print("x2", x2)
        y2 = int(y + w * np.sin(ang))
        x3 = int(x + w * np.cos(ang) - h * np.sin(ang))
        y3 = int(y + w * np.sin(ang) + h * np.cos(ang))
        x4 = int(x - h * np.sin(ang))
        y4 = int(y + h * np.cos(ang))
        color = (30 * i, 0, 255 - 30 * i)
        box = np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]])
        cv.drawContours(img2, [box], 0, color, 1)  # 将旋转矩形视为轮廓绘制
        # cv.line(img2, (x1, y1), (x2, y2), color)
        # cv.line(img2, (x2, y2), (x3, y3), color)
        # cv.line(img2, (x3, y3), (x4, y4), color)
        # cv.line(img2, (x4, y4), (x1, y1), color)

    plt.figure(figsize=(9, 3.2))
    plt.subplot(121), plt.title("(1) img1"), plt.axis('off')
    plt.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
    plt.subplot(122), plt.title("(2) img2"), plt.axis('off')
    plt.imshow(cv.cvtColor(img2, cv.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()

