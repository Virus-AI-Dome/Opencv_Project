"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【1511】运动目标跟踪之密集光流法
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def capMovementOF(prvs, next, tSize=100):
    grayPrvs = cv.cvtColor(prvs, cv.COLOR_BGR2GRAY)  # 转为灰度图
    grayNext = cv.cvtColor(next, cv.COLOR_BGR2GRAY)  # 转为灰度图
    flow = cv.calcOpticalFlowFarneback(grayPrvs, grayNext, None, 0.5, 3, 15, 3, 5, 1.2, 0)  # 计算光流以获取点的新位置
    magFlow, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    # 将光流强度转换为 HSV，提高检测能力
    hsv = np.zeros_like(prvs)  # 为绘制创建掩码图片
    hsv[..., 0] = ang * 180 / np.pi / 2  # 色调范围：0~360°
    hsv[..., 1] = 255  # H 色调/S 饱和度/V 亮度
    hsv[..., 2] = cv.normalize(magFlow, None, 0, 255, cv.NORM_MINMAX)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    draw = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)  # (h,w)
    fgOpen = cv.morphologyEx(draw, cv.MORPH_OPEN, kernel)
    _, fgMask = cv.threshold(fgOpen, 25, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(fgMask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # 查找轮廓
    cnts = sorted(contours, key=cv.contourArea, reverse=True)  # 所有轮廓按面积排序
    frameRect = next.copy()
    for cnt in cnts:
        if cv.contourArea(cnt) > tSize:
            (x, y, w, h) = cv.boundingRect(cnt)  # 该函数计算矩形的边界框
            cv.rectangle(frameRect, (x,y), (x+w,y+h), (0,0,255), 3)
        else:
            break
    return magFlow, frameRect

if __name__ == '__main__':
    # 创建视频读取/捕获对象
    vedioRead = "../images/Vid02.mp4"  # 读取视频文件的路径
    videoCap = cv.VideoCapture(vedioRead)  # 实例化 VideoCapture 类
    ret, frameNew = videoCap.read()  # 读取第一帧图像

    frameNum = 0  # 视频帧数初值
    tf = 120  # 设置抽帧间隔
    plt.figure(figsize=(9, 5.6))
    while videoCap.isOpened():  # 检查视频捕获是否成功
        frameOld = frameNew.copy()
        ret, frameNew = videoCap.read()  # 读取一帧图像
        if ret is True:
            frameNum += 1  # 读取视频的帧数
            magFlow, frameCap = capMovementOF(frameOld, frameNew, tSize=200)
            print(frameNum, magFlow.shape)
            cv.imshow('frame', frameOld)  # 原始视频
            cv.imshow('capture', frameCap)  # 目标识别视频
            cv.imshow('flow', magFlow)  # 光流矩阵
            if (frameNum%tf==0 and 1<=frameNum//tf<=3):  # 判断抽帧条件
                plt.subplot(2, 3, frameNum//tf), plt.axis('off')
                plt.title("({}) Target tracking (f={})".format(frameNum//tf, frameNum))
                plt.imshow(cv.cvtColor(frameCap[:, 300:1200], cv.COLOR_BGR2RGB))
                plt.subplot(2, 3, 3+frameNum//tf), plt.axis('off')
                plt.title("({}) Optical flow (f={})".format(3+frameNum//tf, frameNum))
                plt.imshow(magFlow[:, 300:1200], cmap='gray')
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

