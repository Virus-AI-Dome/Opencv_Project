"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【0206】图像的马赛克处理
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    filepath = "../images/Lena.tif"  # 读取文件的路径
    img = cv.imread(filepath, flags=1)  # flags=1 读取彩色图像(BGR)

    # roi = cv.selectROI(img, showCrosshair=True, fromCenter=False)
    # x, y, wRoi, hRoi = roi  # 矩形裁剪区域的位置参数
    x, y, wRoi, hRoi = 208, 176, 155, 215  # 矩形裁剪区域
    imgROI = img[y:y+hRoi, x:x+wRoi].copy()  # 切片获得矩形裁剪区域
    print(x, y, wRoi, hRoi)

    plt.figure(figsize=(9, 6))
    plt.subplot(231), plt.title("(1) Original"), plt.axis('off')
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.subplot(232), plt.title("(2) Region of interest"), plt.axis('off')
    plt.imshow(cv.cvtColor(imgROI, cv.COLOR_BGR2RGB))

    mosaic = np.zeros(imgROI.shape, np.uint8)  # ROI 区域
    ksize = [5, 10, 20]  # 马赛克网格块的宽度
    for i in range(3):
        k = ksize[i]
        for h in range(0, hRoi, k):
            for w in range(0, wRoi, k):
                color = imgROI[h,w]
                mosaic[h:h+k,w:w+k,:] = color  # 用顶点颜色覆盖马赛克网格块
        imgMosaic = img.copy()
        imgMosaic[y:y + hRoi, x:x + wRoi] = mosaic
        plt.subplot(2,3,i+4), plt.title("({}) Coding image (size={})".format(i+4, k)), plt.axis('off')
        plt.imshow(cv.cvtColor(imgMosaic, cv.COLOR_BGR2RGB))

    plt.subplot(233), plt.title("(3) Mosaic"), plt.axis('off')
    plt.imshow(cv.cvtColor(mosaic, cv.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()
