"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【1510】运动目标跟踪之背景差分法
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    # 创建视频读取/捕获对象
    vedioRead = "../images/Vid02.mp4"  # 读取视频文件的路径
    videoCap = cv.VideoCapture(vedioRead)  # 实例化 VideoCapture 类

    # 混合高斯模型 (GMM) 背景建模方法
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))  # 滤波器核
    backModel = cv.createBackgroundSubtractorMOG2()  # 创建 GMM 模型
    frameNum = 0  # 视频帧数初值
    timef = 60  # 设置抽帧间隔
    plt.figure(figsize=(9, 5.6))
    while videoCap.isOpened():  # 检查视频捕获是否成功
        ret, frame = videoCap.read()  # 读取一帧图像
        if ret is True:
            frameNum += 1  # 读取视频的帧数
            img = backModel.apply(frame)  # 背景建模
            if frameNum > 50:
                # 开运算过滤噪声
                imgClose = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
                # 查找轮廓，只取最外层
                contours, hierarchy = cv.findContours(imgClose, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                cnts = sorted(contours, key=cv.contourArea, reverse=True)  # 所有轮廓按面积排序
                for cnt in cnts:
                    area = cv.contourArea(cnt)  # 计算轮廓面积
                    if area > 200:  # 忽略小目标
                        x, y, w, h = cv.boundingRect(cnt)
                        cv.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 3)
                    else:
                        break
                cv.imshow('frame', frame)  # 目标识别视频
                cv.imshow('img', img)  # 高斯模型视频
                if (frameNum%timef==0 and 1<=frameNum//60<=6):  # 判断抽帧条件
                    plt.subplot(2, 3, frameNum//60), plt.axis('off')
                    plt.title("({}) Target tracking (f={})".format(frameNum//60, frameNum))
                    plt.imshow(cv.cvtColor(frame[:, 300:1200], cv.COLOR_BGR2RGB))
                if cv.waitKey(10) & 0xFF == 27:  # 按 'Esc' 退出
                    break
        else:
            print("Can't receive frame at frameNum {}.".format(frameNum))
            break

    # 释放资源
    videoCap.release()  # 关闭读取视频文件
    cv.destroyAllWindows()  # 关闭显示窗口
    plt.tight_layout()
    plt.show()
