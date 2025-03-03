"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【0404】绘制圆形
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = np.ones((400, 600, 3), np.uint8)*192

    center = (0, 0)  # 圆心坐标 (x,y)
    cx, cy = 300, 200  # 圆心坐标 (x,y)
    for r in range(200, 0, -20):
        color = (r, r, 255-r)
        cv.circle(img, (cx, cy), r, color, -1)  # -1 表示内部填充
        cv.circle(img, center, r, 255)
        cv.circle(img, (600,400), r, color, 5)  # 线宽为 5

    plt.figure(figsize=(6, 4))
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.show()

