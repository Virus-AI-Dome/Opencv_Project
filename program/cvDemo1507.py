"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【1507】框选前景实现图割算法
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv.imread("../images/Fig1502.png", flags=1)  # 读取彩色图像(BGR)
    img= img[:,:500,:]
    mask = np.zeros(img.shape[:2], np.uint8)
    print(img.shape[:2])

    # 定义矩形框，框选目标前景
    # rect = (118, 125, 220, 245)  # 直接设置矩形的位置参数，也可以鼠标框选 ROI
    print("Select a ROI and then press SPACE or ENTER button!\n")
    roi = cv.selectROI(img, showCrosshair=True, fromCenter=False)
    xmin, ymin, w, h = roi  # 矩形裁剪区域 (ymin:ymin+h, xmin:xmin+w) 的位置参数
    rect = (xmin, ymin, w, h)  # 边界框矩形的坐标和尺寸
    imgROI = np.zeros_like(img)  # 创建与 image 相同形状的黑色图像
    imgROI[ymin:ymin+h, xmin:xmin+w] = img[ymin:ymin+h, xmin:xmin+w].copy()
    print(xmin, ymin, w, h)

    fgModel = np.zeros((1, 65), dtype="float")  # 前景模型, 13*5
    bgModel = np.zeros((1, 65), dtype="float")  # 背景模型, 13*5
    iter = 5
    (mask, bgModel, fgModel) = cv.grabCut(img, mask, rect, bgModel, fgModel, iter,
                               mode=cv.GC_INIT_WITH_RECT)  # 框选前景分割模式

    # 将所有确定背景和可能背景像素设置为 0，而确定前景和可能前景像素设置为 1
    maskOutput = np.where((mask==cv.GC_BGD) | (mask==cv.GC_PR_BGD), 0, 1)
    maskGrabCut = (maskOutput * 255).astype("uint8")
    imgGrabCut = cv.bitwise_and(img, img, mask=maskGrabCut)

    plt.figure(figsize=(9, 5.5))
    plt.subplot(231), plt.axis('off'), plt.title("(1) Original")
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))  # 显示 img(RGB)
    plt.subplot(232), plt.axis('off'), plt.title("(2) Bounding box")
    plt.imshow(cv.cvtColor(imgROI, cv.COLOR_BGR2RGB))  # 显示 img(RGB)
    plt.subplot(233), plt.axis('off'), plt.title("(3) definite background")
    maskBGD = np.uint8((mask==cv.GC_BGD)) * 205
    plt.imshow(maskBGD, cmap='gray', vmin=0, vmax=255)  # definite background
    plt.subplot(234), plt.axis('off'), plt.title("(4) probable background")
    maskPBGD = np.uint8((mask==cv.GC_PR_BGD)) * 205
    plt.imshow(maskPBGD, cmap='gray', vmin=0, vmax=255)  # probable background
    plt.subplot(235), plt.axis('off'), plt.title("(5) GrabCut Mask")
    # maskGrabCut = np.where((mask==cv.GC_BGD) | (mask==cv.GC_PR_BGD), 0, 1)
    plt.imshow(maskGrabCut, 'gray')  # mask generated by GrabCut
    plt.subplot(236), plt.axis('off'), plt.title("(6) GrabCut Output")
    plt.imshow(cv.cvtColor(imgGrabCut, cv.COLOR_BGR2RGB))  # GrabCut Output
    plt.tight_layout()
    plt.show()

