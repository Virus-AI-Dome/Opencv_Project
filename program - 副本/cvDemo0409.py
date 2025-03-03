"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【0409】鼠标交互获取多边形区域
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def onMouseAction(event, x, y, flags, param):  # 鼠标交互 (左键选点右键完成)
    global pts
    setpoint = (x, y)
    if event == cv.EVENT_LBUTTONDOWN:  # 鼠标左键点击
        pts.append(setpoint)  # 选中一个多边形顶点
        print("选择顶点 {}：{}".format(len(pts), setpoint))
    elif event == cv.EVENT_MBUTTONDOWN:  # 鼠标中键点击
        pts.pop()  # 取消最近一个顶点
    elif event == cv.EVENT_RBUTTONDOWN:  # 鼠标右键点击
        param = False  # 结束绘图状态
        print("结束绘制，按 ESC 退出。")

if __name__ == '__main__':
    img = cv.imread("../images/Lena.tif")  # 读取彩色图像(BGR)
    imgCopy = img.copy()

    # 鼠标交互 ROI
    print("单击左键：选择 ROI 顶点")
    print("单击中键：删除最近的顶点")
    print("单击右键：结束 ROI 选择")
    print("按 ESC 退出")
    pts = []  # 初始化 ROI 顶点坐标集合
    status = True  # 开始绘图状态
    cv.namedWindow('origin')  # 创建图像显示窗口
    cv.setMouseCallback('origin', onMouseAction, status)  # 绑定回调函数
    while True:
        if len(pts) > 0:
            cv.circle(imgCopy, pts[-1], 5, (0,0,255), -1)  # 绘制最近一个顶点
        if len(pts) > 1:
            cv.line(imgCopy, pts[-1], pts[-2], (255, 0, 0), 2)  # 绘制最近一段线段
        if status == False:  # 判断结束绘制 ROI
            cv.line(imgCopy, pts[0], pts[-1], (255,0,0), 2)  # 绘制最后一段线段
        cv.imshow('origin', imgCopy)
        key = 0xFF & cv.waitKey(10)  # 按 ESC 退出
        if key == 27:  # Esc 退出
            break
    cv.destroyAllWindows()  # 释放图像窗口

    # 提取多边形 ROI
    print("ROI 顶点坐标：", pts)
    points = np.array(pts)  # ROI 多边形顶点坐标集
    cv.polylines(img, [points], True, (255,255,255), 2)  # 在 img 绘制 ROI 多边形
    mask = np.zeros(img.shape[:2], np.uint8)  # 黑色掩模，单通道
    cv.fillPoly(mask, [points], (255,255,255))  # 多边形 ROI 为白色窗口
    imgROI = cv.bitwise_and(img, img, mask=mask)  # 按位与，从 img 中提取 ROI

    plt.figure(figsize=(9, 3.5))
    plt.subplot(131), plt.title("(1) Original"), plt.axis('off')
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.subplot(132), plt.title("(2) ROI mask"), plt.axis('off')
    plt.imshow(mask, cmap='gray')
    plt.subplot(133), plt.title("(3) ROI cropped"), plt.axis('off')
    plt.imshow(cv.cvtColor(imgROI, cv.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()


