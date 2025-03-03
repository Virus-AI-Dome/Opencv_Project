"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""


# 【1609】特征描述之 HOG 描述符
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def drawHOG(image, descriptors, cx, cy, rad):
    angles = np.arange(0, 180, 22.5).astype(np.float32)  # start, stop, step
    normGrad = descriptors/np.max(descriptors).astype(np.float32)
    gx, gy = cv.polarToCart(normGrad*rad, angles, angleInDegrees=True)
    for i in range(angles.shape[0]):
        px, py = int(cx+gx[i]), int(cy+gy[i])
        cv.arrowedLine(image, (cx,cy), (px, py), 0, tipLength=0.1)  # 黑色
    return image

if __name__ == '__main__':
    # (1) 读取样本图像，构造样本图像集合
    img = cv.imread("../images/Fig1101.png", flags=0)  # 灰度图像
    height, width, wCell, d = 200, 200, 20, 10
    img = cv.resize(img, (width, height))  # 调整为统一尺寸

    # (2) 构造 HOG 检测器
    winSize = (20, 20)
    blockSize = (20, 20)
    blockStride = (20, 20)
    cellSize = (20, 20)
    nbins = 8
    hog = cv.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    lenHOG = nbins * (blockSize[0]/cellSize[0]) * (blockSize[1]/cellSize[1]) \
            * ((winSize[0]-blockSize[0])/blockStride[0] + 1) \
            * ((winSize[1]-blockSize[1])/blockStride[1] + 1)
    print("length of descriptors:", lenHOG)

    # (3) 计算检测区域的 HOG 描述符
    xt, yt = 80, 80  # 检测区域位置
    cell = img[xt:xt+wCell, yt:yt+wCell]
    cellDes = hog.compute(cell)  # HOG 描述符，(8,)
    normGrad = cellDes/np.max(cellDes).astype(np.float32)
    print("shape of descriptors:{}".format(cellDes.shape))
    print(cellDes)

    # (4) 绘制方向梯度示意图
    imgGrad = cv.resize(cell, (wCell*10, wCell*10), interpolation=cv.INTER_AREA)
    Gx = cv.Sobel(img, cv.CV_32F, 1, 0, ksize=5)  # X 轴梯度 Gx
    Gy = cv.Sobel(img, cv.CV_32F, 0, 1, ksize=5)  # Y 轴梯度 Gy
    magG, angG = cv.cartToPolar(Gx, Gy, angleInDegrees=True)  # 极坐标求幅值与方向 (0~360)
    print(magG.min(), magG.max(), angG.min(), angG.max())
    angCell = angG[xt:xt+wCell, yt:yt+wCell]
    box = np.zeros((4, 2), np.int32)  # 计算旋转矩形的顶点, (4, 2)
    for i in range(wCell):
        for j in range(wCell):
            cx, cy = i*10+d, j*10+d
            rect = ((cx,cy), (8,1), angCell[i,j])  # 旋转矩形类
            box = np.int32(cv.boxPoints(rect))  # 计算旋转矩形的顶点, (4, 2)
            cv.drawContours(imgGrad, [box], 0, (0,0,0), -1)

    # (5) 绘制检测区域的方向梯度直方图
    cellHOG = np.ones((201,201), np.uint8)  # 白色
    cellHOG = drawHOG(cellHOG, cellDes, xt+d, yt+d, 40)

    # (6) 绘制图像的方向梯度直方图
    imgHOG = np.ones(img.shape, np.uint8)*255  # 白色
    for i in range(10):
        for j in range(10):
            xc, yc = 20*i, 20*j
            cell = img[xc:xc+wCell, yc:yc+wCell]
            descriptors = hog.compute(cell)  # HOG 描述符，(8,)
            imgHOG = drawHOG(imgHOG, descriptors, xc+d, yc+d, 8)
    imgWeight = cv.addWeighted(img, 0.5, imgHOG, 0.5, 0)

    plt.figure(figsize=(9, 6.2))
    plt.subplot(231), plt.title("(1) Original")
    cv.rectangle(img, (xt,yt), (xt+wCell,yt+wCell), (0,0,0), 2)  # 绘制 block
    plt.axis('off'), plt.imshow(img, cmap='gray')
    plt.subplot(232), plt.title("(2) Oriented gradient")
    angNorm = np.uint8(cv.normalize(angG, None, 0, 255, cv.NORM_MINMAX))
    plt.axis('off'), plt.imshow(angNorm, cmap='gray')
    plt.subplot(233), plt.title("(3) Image with HOG"), plt.axis('off')
    cv.rectangle(imgWeight, (xt,yt), (xt+wCell,yt+wCell), (0,0,0), 2)  # 绘制 block
    plt.axis('off'), plt.imshow(imgWeight, cmap='gray')
    plt.subplot(234), plt.title("(4) Grad angle of cell")
    plt.axis('off'), plt.imshow(imgGrad, cmap='gray')
    plt.subplot(235), plt.title("(5) HOG of cell")
    strAng = ("0", "22", "45", "67", "90", "112", "135", "157")
    plt.bar(strAng, cellDes*wCell*wCell)
    plt.subplot(236), plt.title("(6) HOG diagram of cell")
    plt.axis('off'), plt.imshow(cellHOG, cmap='gray')
    plt.tight_layout()
    plt.show()
