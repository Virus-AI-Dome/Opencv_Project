"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【1216】形态学重建的粒度分离
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def morphRebuild(F0, template):
    element = cv.getStructuringElement(cv.MORPH_CROSS, (3,3))
    Flast = F0  # F(0)，重建标记
    while True:
        dilateF = cv.dilate(Flast, kernel=element)  # 标记图像膨胀
        Fnew = cv.bitwise_and(dilateF, template)  # 模板约束重建
        if (Fnew==Flast).all():  # 收敛判断，F(k+1)=F(k)?
            break  # 结束迭代
        else:
            Flast = Fnew  # 更新 F(k)
    imgRebuild = Fnew  # 收敛的标记图像 F(k)
    return imgRebuild

if __name__ == '__main__':
    img = cv.imread("../images/Fig1207.png", flags=0)  # 灰度图像
    _, imgBin = cv.threshold(img, 205, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)  # 二值处理 (黑色背景)
    imgBinInv = cv.bitwise_not(imgBin)  # 二值图像的补集 (白色背景)

    # (1) 垂直特征结构元
    element = cv.getStructuringElement(cv.MORPH_RECT, (1,60))  # 垂直特征结构元，高度 60 像素
    imgErode = cv.erode(imgBin, kernel=element)  # 腐蚀结果作为标记图像
    imgRebuild1 = morphRebuild(imgErode, imgBin)  # 形态学重建
    imgDual1 = cv.bitwise_and(imgBin, cv.bitwise_not(imgRebuild1))  # 由 F(k) 的补集获得

    # (2) 水平特征结构元
    element = cv.getStructuringElement(cv.MORPH_RECT, (60,1))  # 水平特征结构元，长度 60 像素
    imgErode = cv.erode(imgBin, kernel=element)  # 腐蚀结果作为标记图像
    imgRebuild2 = morphRebuild(imgErode, imgBin)  # 形态学重建

    # (3) 圆形特征结构元
    element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (60,60))  # 水平特征结构元，长度 60 像素
    imgErode = cv.erode(imgBin, kernel=element)  # 腐蚀结果作为标记图像
    imgRebuild3 = morphRebuild(imgErode, imgBin)  # 形态学重建

    plt.figure(figsize=(9, 6))
    plt.subplot(231), plt.axis("off"), plt.title("(1) Original")
    plt.imshow(img, cmap='gray')
    plt.subplot(232), plt.axis("off"), plt.title("(2) Binary")
    plt.imshow(imgBin, cmap='gray')
    plt.subplot(233), plt.axis("off"), plt.title("(3) Initial marker")
    plt.imshow(imgErode, cmap='gray')
    plt.subplot(234), plt.axis("off"), plt.title("(4) Rebuild1 (1,60)")
    plt.imshow(imgRebuild1, cmap='gray')
    plt.subplot(235), plt.axis("off"), plt.title("(5) Rebuild2 (60,1)")
    plt.imshow(imgRebuild2, cmap='gray')
    plt.subplot(236), plt.axis("off"), plt.title("(6) Rebuild3 (60,60)")
    plt.imshow(imgRebuild3, cmap='gray')
    plt.tight_layout()
    plt.show()