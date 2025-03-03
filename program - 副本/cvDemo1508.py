"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【1508】运动图像分割之均值漂移算法
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    # (1) 创建视频读取/捕获对象
    vedioRead = "../images/Vid01.mp4"  # 读取视频文件的路径
    videoCap = cv.VideoCapture(vedioRead)  # 实例化 VideoCapture 类
    ret, frame = videoCap.read()  # 读取第一帧图像
    height, width = frame.shape[:2]

    # (2) 设置追踪目标窗口
    print("Select a ROI and then press SPACE or ENTER button!\n")
    trackWindow = cv.selectROI(frame, showCrosshair=True, fromCenter=False)
    x, y, w, h = trackWindow
    roiFrame = frame[y:y+h, x:x+w]  # 设置追踪的区域
    plt.figure(figsize=(9, 3.2))
    frame = cv.rectangle(frame[:,:810], (x,y), (x+w,y+h), 255, 2)
    plt.subplot(131), plt.title("(1) Target tracking (f=0)")
    plt.axis('off'), plt.imshow(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
    print("frame.shape: ", frame.shape, roiFrame.shape)

    # (3) 均值漂移算法，在 dst 寻找目标窗口，找到后返回目标窗口位置
    roiHSV = cv.cvtColor(roiFrame, cv.COLOR_BGR2HSV)
    mask = cv.inRange(roiHSV, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
    roiHist = cv.calcHist([roiHSV], [0], mask, [180], [0, 180])  # 计算直方图
    cv.normalize(roiHist, roiHist, 0, 255, cv.NORM_MINMAX)  # 归一化
    termCrit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)
    frameNum = 0  # 视频帧数初值
    timef = 60  # 设置抽帧间隔
    while videoCap.isOpened():  # 检查视频捕获是否成功
        ret, frame = videoCap.read()  # 读取一帧图像
        if ret is True:
            frameNum += 1  # 读取视频的帧数
            print(frameNum)
            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)  # BGR->HSV
            dst = cv.calcBackProject([hsv], [0], roiHist, [0, 180], 1)
            ret, trackWindow = cv.meanShift(dst, trackWindow, termCrit)
            x, y, w, h = trackWindow  # 绘制追踪窗口
            frame = cv.rectangle(frame, (x, y), (x+w, y+h), 255, 2)
            cv.imshow('frameCap', frame)
            if (frameNum%timef==0 and 1<=frameNum//60<=2):  # 判断抽帧条件
                plt.subplot(1, 3, 1+frameNum//60), plt.axis('off')
                plt.title("({}) Target tracking (f={})".format(1+frameNum//60, frameNum))
                plt.imshow(cv.cvtColor(frame[:,:810], cv.COLOR_BGR2RGB))
            if cv.waitKey(10) & 0xFF == 27:  # 按 'Esc' 退出
                break
        else:
            print("Can't receive frame at frameNum {}.".format(frameNum))
            break

    # (4) 释放资源
    videoCap.release()  # 关闭读取视频文件
    cv.destroyAllWindows()  # 关闭显示窗口
    plt.tight_layout()
    plt.show()
