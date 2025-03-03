"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【0908】基于鼠标交互的色彩分割
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def onMouseAction(event, x, y, flags, param):  # 鼠标交互 (左键选点右键完成)
    global pts
    setpoint = (x, y)
    if event == cv.EVENT_LBUTTONDOWN:  # 鼠标左键点击
        pts.append(setpoint)  # 选中一个多边形顶点
        print("{}. 像素点坐标：{}".format(len(pts), setpoint))
    elif event == cv.EVENT_RBUTTONDOWN:  # 鼠标右键点击
        param = False  # 结束绘图状态
        print("结束绘制，按 ESC 退出。")

if __name__ == '__main__':
    # 读取原始图像
    img = cv.imread("../images/Fig0901.png", flags=1)  # 读取彩色图像
    h, w = img.shape[:2]  # 图片的高度, 宽度
    imgCopy = img.copy()
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)  # 将图片转换到 HSV 色彩空间

    # 鼠标交互 ROI
    print("单击左键：选择特征点")
    print("单击右键：结束选择")
    pts = []  # 初始化 ROI 顶点坐标集合
    status = True  # 开始绘图状态
    cv.namedWindow("origin")  # 创建图像显示窗口
    cv.setMouseCallback("origin", onMouseAction, status)  # 绑定回调函数
    while True:
        if len(pts) > 0:
            # cv.circle(imgCopy, pts[-1], 5, (0,0,255), -1)  # 绘制最近一个顶点
            px, py = pts[-1]
            imgROI = img[py-1:py+1, px-1:px+1, :]
            hsvROI = cv.cvtColor(imgROI, cv.COLOR_BGR2HSV)  # 将图片转换到 HSV 色彩空间
            Hmean = hsvROI[:, :, 0].mean()  # 选中区域的色相 H
            Smean = hsvROI[:, :, 1].mean()  # 选中区域的饱和度 S
            if Hmean<=10:
                Hmin, Hmax = 0, 10
            elif 10<Hmean<=25:
                Hmin, Hmax = 11, 25
            elif 25<Hmean<=34:
                Hmin, Hmax = 26, 34
            elif 34<Hmean<=77:
                Hmin, Hmax = 35, 77
            elif 77<Hmean<=99:
                Hmin, Hmax = 78, 99
            elif 99<Hmean<=124:
                Hmin, Hmax = 100, 124
            elif 124<Hmean<=155:
                Hmin, Hmax = 125, 155
            else:
                Hmin, Hmax = 156, 180
            Smin, Smax = 43, 255
            Vmin, Vmax = 46, 255
            if 0<Smean<43:
                Hmin, Hmax = 0, 180
                Smin, Smax = 0, 43
            lower, upper = (Hmin, Smin, Vmin), (Hmax, Smax, Vmax)
            binary = cv.inRange(hsv, lower, upper)  # 选中颜色区域为 1
            binaryInv = cv.bitwise_not(binary)  # 生成逆遮罩，选中颜色区域为 0
            imgCopy[binaryInv==0] = [255, 0, 0]  # 选中颜色区域修改为蓝色
        cv.imshow('origin', imgCopy)
        key = 0xFF & cv.waitKey(10)  # 按 ESC 退出
        if key == 27:  # Esc 退出
            break
    cv.destroyAllWindows()  # 释放图像窗口

    plt.figure(figsize=(9, 3.5))
    plt.subplot(131), plt.title("(1) Original"), plt.axis('off')
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.subplot(132), plt.title("(2) Background"), plt.axis('off')
    plt.imshow(binary, cmap='gray', vmin=0, vmax=255)
    plt.subplot(133), plt.title("(3) Matting"), plt.axis('off')
    plt.imshow(cv.cvtColor(imgCopy, cv.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()
