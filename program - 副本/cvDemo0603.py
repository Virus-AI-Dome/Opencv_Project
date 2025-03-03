"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【0603】图像的旋转
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv.imread("../images/Fig0301.png")  # 读取彩色图像(BGR)
    height, width = img.shape[:2]  # 图片的高度和宽度

    # (1) 以原点为中心旋转
    x0, y0 = 0, 0  # 以左上角顶点 (0,0) 作为旋转中心
    theta, scale = 30, 1.0  # 逆时针旋转 30 度，缩放系数 1.0
    MAR0 = cv.getRotationMatrix2D((x0,y0), theta, scale)  # 旋转变换矩阵
    imgRot1 = cv.warpAffine(img, MAR0, (width, height))  #

    # (2) 以任意点为中心旋转
    x0, y0 = width//2, height//2  # 以图像中心作为旋转中心
    angle = theta * np.pi/180  # 弧度->角度
    wRot = int(width * np.cos(angle) + height * np.sin(angle))  # 调整宽度
    hRot = int(height * np.cos(angle) + width * np.sin(angle))  # 调整高度
    scale = width/wRot  # 根据 wRot 调整缩放系数
    MAR1 = cv.getRotationMatrix2D((x0,y0), theta, 1.0)  # 逆时针旋转 30 度，缩放系数 1.0
    MAR2 = cv.getRotationMatrix2D((x0,y0), theta, scale)  # 逆时针旋转 30 度，缩放比例 scale
    imgRot2 = cv.warpAffine(img, MAR1, (width, height), borderValue=(255,255,255))  # 白色填充
    imgRot3 = cv.warpAffine(img, MAR2, (width, height))  # 调整缩放系数以保留原图内容
    print(img.shape, imgRot2.shape, imgRot3.shape, scale)

    # (3) 图像的直角旋转
    imgRot90 = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)  # 顺时针旋转 90度
    imgRot180 = cv.rotate(img, cv.ROTATE_180)  # 顺时针旋转 180度
    imgRot270 = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)  # 顺时针旋转 270度

    plt.figure(figsize=(9, 6))
    plt.subplot(231), plt.title("(1) Rotate around the origin"), plt.axis('off')
    plt.imshow(cv.cvtColor(imgRot1, cv.COLOR_BGR2RGB))
    plt.subplot(232), plt.title("(2) Rotate around the center"), plt.axis('off')
    plt.imshow(cv.cvtColor(imgRot2, cv.COLOR_BGR2RGB))
    plt.subplot(233), plt.title("(3) Rotate and resize"), plt.axis('off')
    plt.imshow(cv.cvtColor(imgRot3, cv.COLOR_BGR2RGB))
    plt.subplot(234), plt.title("(4) Rotate 90 degrees"), plt.axis('off')
    plt.imshow(cv.cvtColor(imgRot90, cv.COLOR_BGR2RGB))
    plt.subplot(235), plt.title("(5) Rotate 180 degrees"), plt.axis('off')
    plt.imshow(cv.cvtColor(imgRot180, cv.COLOR_BGR2RGB))
    plt.subplot(236), plt.title("(6) Rotate 270 degrees"), plt.axis('off')
    plt.imshow(cv.cvtColor(imgRot270, cv.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()





