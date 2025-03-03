"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【0409】鼠标交互获取多边形区域
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def onMouseAction(event,x,y,flags,param):
    global pts
    setpoint = (x,y)

    if event == cv.EVENT_LBUTTONDOWN:
        pts.append(setpoint)
        print("选择顶点：{}，{}".format(len(pts),setpoint))
    elif event == cv.EVENT_MBUTTONDOWN:
        pts.pop()
    elif event == cv.EVENT_RBUTTONDOWN:
        param =False
        print("绘制结束")


if __name__ == '__main__':

    img = cv.imread("../images/Lena.tif")
    imgCopy = img.copy()
    pts = []
    status = True
    cv.namedWindow('origin')
    cv.setMouseCallback('origin',onMouseAction,status)
    while True:
        if len(pts) >0:
            cv.circle(imgCopy,pts[-1],5,(0,0,255),-1)
        if len(pts) >1:
            cv.line(imgCopy,pts[-1],pts[-2],(255,0,0),2)
        if status == False:
            cv.line(imgCopy,pts[0],pts[-1],(255,0,0),2)
        cv.imshow('origin',imgCopy)
        key = 0xFF & cv.waitKey(10)
        if key ==27 :
           break
    cv.destroyAllWindows()

    # 提取 ROI 区域
    print("ROI 顶点坐标：", pts)
    points = np.array(pts)
    cv.polylines(img,[points],True,(255,255,255),2)
    mask = np.zeros(img.shape[:2],np.uint8)
    cv.fillPoly(mask,[points],(255,255,255))
    imgROI = cv.bitwise_and(img,img,mask=mask)

    plt.figure(figsize=(9,6))
    plt.subplot(1,3,1)
    plt.title('(1) origin')
    plt.axis('off')
    plt.imshow(cv.cvtColor(img,cv.COLOR_BGR2RGB))
    plt.subplot(1,3,2)
    plt.title('(2) ROI MASK')
    plt.axis('off')
    plt.imshow(cv.cvtColor(mask,cv.COLOR_BGR2RGB))
    plt.subplot(1,3,3)
    plt.title('(3) ROI cropped')
    plt.axis('off')
    plt.imshow(cv.cvtColor(imgROI,cv.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()



