"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【0401】绘制直线与线段
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    height, width, channels = 180, 200, 3
    img = np.ones((height, width, channels), np.uint8) * 160  # 创建灰色图像

    # (1) 线条参数 color 的设置
    # 注意 pt1、pt2 坐标格式是 (x,y) 而不是 (y,x)
    img1 = img.copy()  # 绘图函数就地操作，会修改输入图像
    cv.line(img1, (0,0), (200,180), (0,0,255), 1)  # 红色 R=255
    cv.line(img1, (0,0), (100,180), (0,255,0), 1)  # 绿色 G=255
    cv.line(img1, (0,40), (200,40), (128,0,0), 2)  # 深蓝 B=128
    cv.line(img1, (0,80), (200,80), 128, 2)  # color=128 等效于 (128,0,0)
    cv.line(img1, (0,120), (200,120), 255, 2)  # color=255 等效于 (255,0,0)

    # (2) 线宽的设置
    # 如果设置了 thickness，关键词 "lineType" 可以省略
    img2 = img.copy()
    cv.line(img2, (20,50), (180,10), (255,0,0), 1, cv.LINE_8)  # 绿色
    cv.line(img2, (20,90), (180,50), (255,0,0), 1, cv.LINE_AA)  # 绿色
    # 如果没有设置 thickness，则关键词 "lineType" 不能省略
    cv.line(img2, (20,130), (180,90), (255,0,0), cv.LINE_8)  # 蓝色, cv.LINE 被识别为线宽
    cv.line(img2, (20,170), (180,130), (255,0,0), cv.LINE_AA)  # 蓝色, cv.LINE 被识别为线宽

    # (3) tipLength 指箭头部分长度与整个线段长度的比例
    img3 = img.copy()
    img3 = cv.arrowedLine(img3, (20,20), (180,20), (0,0,255), tipLength=0.05)  # 从 pt1 指向 pt2
    img3 = cv.arrowedLine(img3, (20,60), (180,60), (0,0,255), tipLength=0.1)
    img3 = cv.arrowedLine(img3, (20,100), (180,100), (0,0,255), tipLength=0.15)  # 双向箭头
    img3 = cv.arrowedLine(img3, (180,100), (20,100), (0,0,255), tipLength=0.15)  # 交换 pt1、pt2
    img3 = cv.arrowedLine(img3, (20,140), (210,140), (0,0,255), tipLength=0.2)  # 终点越界，箭头未显示

    # (4) 没有复制原图，将直接改变输入图像 img，可能导致相互影响
    img4 = cv.line(img, (0,100), (150,100), (0,255,0), 1)  # 水平线, y=100
    img5 = cv.line(img, (75,0), (75,200), (0,0,255), 1)  # 垂直线, x= 60

    # (5) 灰度图像上只能绘制灰度线条，参数 color 只有第一通道值有效
    img6 = np.zeros((height, width), np.uint8)  # 创建灰度图像
    cv.line(img6, (0,10), (200,10), (0,255,255), 2)  # Gray=0
    cv.line(img6, (0,30), (200,30), (64,128,255), 2)  # Gray=64
    cv.line(img6, (0,60), (200,60), (128,64,255), 2)  # Gray=128
    cv.line(img6, (0,100), (200,100), (255,0,255), 2)  # Gray=255
    cv.line(img6, (20,0), (20,200), 128, 2)  # Gray=128
    cv.line(img6, (60,0), (60,200), (255,0,0), 2)  # Gray=255
    cv.line(img6, (100,0), (100,200), (255,255,255), 2)  # Gray=255
    print(img6.shape, img6.shape)

    plt.figure(figsize=(9, 6))
    plt.subplot(231), plt.title("(1) img1"), plt.axis('off')
    plt.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
    plt.subplot(232), plt.title("(2) img2"), plt.axis('off')
    plt.imshow(cv.cvtColor(img2, cv.COLOR_BGR2RGB))
    plt.subplot(233), plt.title("(3) img3"), plt.axis('off')
    plt.imshow(cv.cvtColor(img3, cv.COLOR_BGR2RGB))
    plt.subplot(234), plt.title("(4) img4"), plt.axis('off')
    plt.imshow(cv.cvtColor(img4, cv.COLOR_BGR2RGB))
    plt.subplot(235), plt.title("(5) img5"), plt.axis('off')
    plt.imshow(cv.cvtColor(img5, cv.COLOR_BGR2RGB))
    plt.subplot(236), plt.title("(6) img6"), plt.axis('off')
    plt.imshow(img6, cmap="gray")
    plt.tight_layout()
    plt.show()
