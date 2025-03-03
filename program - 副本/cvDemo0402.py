"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【0402】绘制垂直的矩形
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    height, width, channels = 300, 320, 3
    img = np.ones((height, width, channels), np.uint8) * 192  # 创建灰色图像

    # (1) 矩形参数的设置 Pt1(x1,y1), Pt2(x2,y2)
    img1 = img.copy()
    cv.rectangle(img1, (0,80), (100,220), (0,0,255), 2)  # 比较 (x,y) 与 (y,x)
    cv.rectangle(img1, (80,0), (220,100), (0,255,0), 2)  # (y,x)
    cv.rectangle(img1, (150,120), (400,200), 255, 2)  # 越界自动裁剪
    cv.rectangle(img1, (50,10), (100,50), (128,0,0), 1)  # 线宽的影响
    cv.rectangle(img1, (150,10), (200,50), (192,0,0), 2)
    cv.rectangle(img1, (250,10), (300,50), (255,0,0), 4)
    cv.rectangle(img1, (50,250), (100,290), (128,0,0), -1)  # 内部填充
    cv.rectangle(img1, (150,250), (200,290), (192,0,0), -1)
    cv.rectangle(img1, (250,250), (300,290), (255,0,0), -1)

    # (2) 通过 (x, y, w, h) 绘制矩形
    img2 = img.copy()
    x, y, w, h = (50, 100, 200, 100)  # 左上角坐标 (x,y), 宽度 w，高度 h
    cv.rectangle(img2, (x, y), (x+w, y+h), (0,0,255), 2)
    text = "({},{}),{}*{}".format(x, y, w, h)
    cv.putText(img2, text, (x,y-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))

    # (3) 在灰度图像中绘制直线和矩形
    img3 = np.zeros((height, width), np.uint8)  # 创建黑色背景图像
    cv.line(img3, (0,40), (320,40), 64, 2)
    cv.line(img3, (0,80), (320,80), (128,128,255), 2)
    cv.line(img3, (0,120), (320,120), (192,64,255), 2)
    cv.rectangle(img3, (20,250), (50,220), 128, -1)  # Gray=128
    cv.rectangle(img3, (80,250), (110,210), (128,0,0), -1)  # Gray=128
    cv.rectangle(img3, (140,250), (170,200), (128,255,255), -1)  # Gray=128
    cv.rectangle(img3, (200,250), (230,190), 192, -1)  # Gray=192
    cv.rectangle(img3, (260,250), (290,180), 255, -1)  # Gray=255

    plt.figure(figsize=(9, 3.3))
    plt.subplot(131), plt.title("(1) img1"), plt.axis('off')
    plt.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
    plt.subplot(132), plt.title("(2) img2"), plt.axis('off')
    plt.imshow(cv.cvtColor(img2, cv.COLOR_BGR2RGB))
    plt.subplot(133), plt.title("(3) img3"), plt.axis('off')
    plt.imshow(img3, cmap="gray")
    plt.tight_layout()
    plt.show()

