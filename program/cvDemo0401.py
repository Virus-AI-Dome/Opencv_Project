"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【0401】绘制直线与线段
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':

    height,wight,channels = 180,200,3
    img =np.ones((height,wight,channels),np.uint8)*160


    img1 = img.copy()
    # 画直线
    cv.line(img1,(0,0),(200,180),(0,0,255),2)
    cv.line(img1,(0,0),(100,180),(0,255,0),2)
    cv.line(img1,(0,40),(200,40),(128,0,0),2)
    cv.line(img1,(0,80),(200,80),(128,0,0),2)
    cv.line(img1,(0,100),(200,100),(128,0,0),2)

    # 线宽
    img2 = img.copy()
    cv.line(img2, (20, 50), (180, 10), (255, 0, 0), 1, cv.LINE_8)  # 绿色
    cv.line(img2, (20, 90), (180, 50), (255, 0, 0), 1, cv.LINE_AA)  # 绿色
    # 如果没有设置 thickness，则关键词 "lineType" 不能省略
    cv.line(img2, (20, 130), (180, 90), (255, 0, 0), cv.LINE_8)  # 蓝色, cv.LINE 被识别为线宽
    cv.line(img2, (20, 170), (180, 130), (255, 0, 0), cv.LINE_AA)  # 蓝色, cv.LINE 被识别为线宽

    # 箭头直线
    img3 = img.copy()
    img3 = cv.arrowedLine(img3,(20,50),(180,10),(255,0,0),tipLength=0.05)
    img3 = cv.arrowedLine(img3,(20,90),(180,50),(0,0,255),tipLength=0.15)

    # 灰度图

    img6 = np.zeros((180,200),np.uint8)





    images = [img1,img2,img3,img6]
    plt.figure(figsize=(9,6))
    for i in range(4):
        plt.subplot(2,3,i+1)
        plt.axis('off')
        plt.title('({}) image'.format(i+1))
        plt.imshow(cv.cvtColor(images[i],cv.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()


