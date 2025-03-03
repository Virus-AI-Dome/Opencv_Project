"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【0405】在图像上绘制椭圆
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = np.ones((400, 600, 3), np.uint8)*224
    img1 = img.copy()
    img2 = img.copy()

    # (1) 半轴长度 (haf) 的影响
    cx, cy = 150, 200  # 圆心坐标
    angle = 120  # 旋转角度
    startAng, endAng = 0, 360  # 开始角度，结束角度
    haf = [50, 100, 150, 180]  # 第一轴的半轴长度
    has = 100  # 第二轴的半轴长度
    for i in range(len(haf)):
        color = (i*50, i*50, 255-i*50)
        cv.ellipse(img1, (cx,cy), (haf[i],has), angle, startAng, endAng, color, 2)
        angPi = angle * np.pi / 180  # 转换为弧度制，便于计算坐标
        xe = int(cx + haf[i]*np.cos(angPi))
        ye = int(cy + haf[i]*np.sin(angPi))
        cv.circle(img1, (xe,ye), 2, color, -1)
        cv.arrowedLine(img1, (cx,cy), (xe,ye), color)  # 从圆心指向第一轴端点
        text = "haF={}".format(haf[i])
        cv.putText(img1, text, (xe+5,ye), cv.FONT_HERSHEY_SIMPLEX, 0.6, color)
    # 绘制第二轴
    xe = int(cx + has*np.sin(angPi))  # 计算第二轴端点坐标
    ye = int(cy - has*np.cos(angPi))
    cv.arrowedLine(img1, (cx, cy), (xe, ye), color)  # 从圆心指向第二轴端点
    text = "haS={}".format(has)
    cv.putText(img1, text, (xe-80, ye+30), cv.FONT_HERSHEY_SIMPLEX, 0.6, color)

    # (2) 旋转角度 (angle) 的影响
    cx, cy = 420, 180  # 圆心坐标
    haf, has = 120, 60  # 半轴长度
    startAng, endAng = 0,360  # 开始角度，结束角度
    angle = [0, 30, 60, 135]  # 旋转角度
    for i in range(len(angle)):
        color = (i*50, i*50, 255-i*50)
        cv.ellipse(img1, (cx,cy), (haf,has), angle[i], startAng, endAng, color, 2)
        angPi = angle[i] * np.pi / 180  # 转换为弧度制，便于计算坐标
        xe = int(cx + haf*np.cos(angPi))
        ye = int(cy + haf*np.sin(angPi))
        cv.circle(img1, (xe,ye), 2, color, -1)
        cv.arrowedLine(img1, (cx,cy), (xe,ye), color)  # 从圆心指向第一轴端点
        text = "rot {}".format(angle[i])
        cv.putText(img1, text, (xe+5,ye), cv.FONT_HERSHEY_SIMPLEX, 0.6, color)

    # (3) 起始角度 (startAngle) 的影响 I
    cx, cy = 60, 80  # 圆心坐标
    haf, has = 45, 30  # 半轴长度
    angle = 0  # 旋转角度
    endAng = 360  # 结束角度
    startAng = [0, 45, 90, 180]  # 开始角度
    for i in range(len(startAng)):
        color = (i*20, i*20, 255-i*20)
        cyi = cy+i*90
        cv.ellipse(img2, (cx,cyi), (haf,has), angle, startAng[i], endAng, color, 2)
        angPi = angle * np.pi / 180  # 转换为弧度制，便于计算坐标
        xe = int(cx + haf*np.cos(angPi))
        ye = int(cyi + haf*np.sin(angPi))
        cv.arrowedLine(img2, (cx,cyi), (xe,ye), 255)  # 从圆心指向第一轴端点
        text = "start {}".format(startAng[i])
        cv.putText(img2, text, (cx-40,cyi), cv.FONT_HERSHEY_SIMPLEX, 0.6, color)
    text = "end={}".format(endAng)
    cv.putText(img2, text, (10, cy-50), cv.FONT_HERSHEY_SIMPLEX, 0.6, 255)

    # (4) 起始角度 (startAngle) 的影响 II
    cx, cy = 180, 80  # 圆心坐标
    haf, has = 45, 30  # 半轴长度
    angle = 30  # 旋转角度
    endAng = 360  # 结束角度
    startAng = [0, 45, 90, 180]  # 开始角度
    for i in range(len(startAng)):
        color = (i*20, i*20, 255-i*20)
        cyi = cy+i*90
        cv.ellipse(img2, (cx,cyi), (haf,has), angle, startAng[i], endAng, color, 2)
        angPi = angle * np.pi / 180  # 转换为弧度制，便于计算坐标
        xe = int(cx + haf*np.cos(angPi))
        ye = int(cyi + haf*np.sin(angPi))
        cv.arrowedLine(img2, (cx,cyi), (xe,ye), 255)  # 从圆心指向第一轴端点
        text = "start {}".format(startAng[i])
        cv.putText(img2, text, (cx-40,cyi), cv.FONT_HERSHEY_SIMPLEX, 0.6, color)
    text = "end={}".format(endAng)
    cv.putText(img2, text, (150,cy-50), cv.FONT_HERSHEY_SIMPLEX, 0.6, 255)

    # (5) 结束角度 (endAngle) 的影响 I
    cx, cy = 300, 80  # 圆心坐标
    haf, has = 45, 30  # 半轴长度
    angle = 0  # 旋转角度
    startAng = 0  # 开始角度
    endAng = [45, 90, 180, 360]  # 结束角度
    for i in range(len(endAng)):
        color = (i*20, i*20, 255-i*20)
        cyi = cy+i*90
        cv.ellipse(img2, (cx,cyi), (haf,has), angle, startAng, endAng[i], color, 2)
        angPi = angle * np.pi / 180  # 转换为弧度制，便于计算坐标
        xe = int(cx + haf*np.cos(angPi))
        ye = int(cyi + haf*np.sin(angPi))
        cv.arrowedLine(img2, (cx,cyi), (xe,ye), 255)  # 从圆心指向第一轴端点
        text = "end {}".format(endAng[i])
        cv.putText(img2, text, (cx-40,cyi), cv.FONT_HERSHEY_SIMPLEX, 0.6, color)
    text = "start={}".format(startAng)
    cv.putText(img2, text, (250,cy-50), cv.FONT_HERSHEY_SIMPLEX, 0.6, 255)

    # (6) 结束角度 (endAngle) 的影响 II
    cx, cy = 420, 80  # 圆心坐标
    haf, has = 45, 30  # 半轴长度
    angle = 30  # 旋转角度
    startAng = 45  # 开始角度
    endAng = [30, 90, 180, 360]  # 结束角度
    for i in range(len(endAng)):
        color = (i*20, i*20, 255-i*20)
        cyi = cy+i*90
        cv.ellipse(img2, (cx,cyi), (haf,has), angle, startAng, endAng[i], color, 2)
        angPi = angle * np.pi / 180  # 转换为弧度制，便于计算坐标
        xe = int(cx + haf*np.cos(angPi))
        ye = int(cyi + haf*np.sin(angPi))
        cv.arrowedLine(img2, (cx,cyi), (xe,ye), 255)  # 从圆心指向第一轴端点
        text = "end {}".format(endAng[i])
        cv.putText(img2, text, (cx-40,cyi), cv.FONT_HERSHEY_SIMPLEX, 0.6, color)
    text = "start={}".format(startAng)
    cv.putText(img2, text, (370,cy-50), cv.FONT_HERSHEY_SIMPLEX, 0.6, 255)

    # (7) 起始角度和结束角度的影响
    cx, cy = 540, 80  # 圆心坐标
    haf, has = 40, 30  # 半轴长度
    angle = 30  # 旋转角度
    startAng = [0, 0, 180, 180 ]  # 开始角度
    endAng = [90, 180, 270, 360]  # 结束角度
    for i in range(len(endAng)):
        color = (i*20, i*20, 255-i*20)
        cyi = cy+i*90
        cv.ellipse(img2, (cx,cyi), (haf,has), angle, startAng[i], endAng[i], color, 2)
        angPi = angle * np.pi / 180  # 转换为弧度制，便于计算坐标
        xe = int(cx + haf*np.cos(angPi))
        ye = int(cyi + haf*np.sin(angPi))
        cv.arrowedLine(img2, (cx,cyi), (xe,ye), 255)  # 从圆心指向第一轴端点
        text = "start {}".format(startAng[i])
        cv.putText(img2, text, (cx-40,cyi-20), cv.FONT_HERSHEY_SIMPLEX, 0.6, color)
        text = "end {}".format(endAng[i])
        cv.putText(img2, text, (cx-40,cyi), cv.FONT_HERSHEY_SIMPLEX, 0.6, color)
    text = "rotate={}".format(angle)
    cv.putText(img2, text, (490,cy-50), cv.FONT_HERSHEY_SIMPLEX, 0.6, 255)

    plt.figure(figsize=(9, 3.5))
    plt.subplot(121), plt.title("(1) Ellipse1"), plt.axis('off')
    plt.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
    plt.subplot(122), plt.title("(2) Ellipse2"), plt.axis('off')
    plt.imshow(cv.cvtColor(img2, cv.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()


