"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【1212】形态学重建之边界清除
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv.imread("../images/Fig1205.png", flags=0)  # 灰度图像
    _, imgBin = cv.threshold(img, 205, 255, cv.THRESH_BINARY_INV)  # 二值处理 (黑色背景)
    imgBinInv = cv.bitwise_not(imgBin)  # 二值图像的补集 (白色背景)，用于构造标记图像

    # 构造标记图像:
    F0 = np.zeros(img.shape, np.uint8)  # 边界为 imgBin，其它全黑
    F0[:, 0] = imgBin[:, 0]
    F0[:, -1] = imgBin[:, -1]
    F0[0, :] = imgBin[0, :]
    F0[-1, :] = imgBin[-1, :]

    # 形态学重建
    Flast = F0.copy()  # F(k) 初值
    element = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    iter = 0
    while True:
        dilateF = cv.dilate(Flast, kernel=element)  # 标记图像膨胀
        Fnew = cv.bitwise_and(dilateF, imgBin)  # 原图像作为模板约束重建
        if (Fnew == Flast).all():  # 收敛判断 F(k+1)=F(k)?
            break  # 结束迭代，Fnew 是收敛的标记图像
        else:
            Flast = Fnew.copy()  # 更新 F(k)
        iter += 1  # 迭代次数
        if iter == 5:
            imgF1 = Fnew  # 显示中间结果
        elif iter == 50:
            imgF50 = Fnew  # 显示中间结果
    print("iter=", iter)
    imgRebuild = cv.bitwise_and(imgBin, cv.bitwise_not(Fnew))  # 计算边界清除后的图像

    plt.figure(figsize=(9, 5.6))
    plt.subplot(231), plt.axis("off"), plt.title("(1) Original")
    plt.imshow(img, cmap='gray')
    plt.subplot(232), plt.axis("off"), plt.title(r"(2) Template ($I^c$)")
    plt.imshow(imgBin, cmap='gray')  # 黑色背景
    plt.subplot(233), plt.axis("off"), plt.title("(3) Initial marker")
    plt.imshow(imgF1, cmap='gray')  # 初始标记图像
    plt.subplot(234), plt.axis("off"), plt.title("(4) Marker (iter=50)")
    plt.imshow(imgF50, cmap='gray')  # 迭代标记图像
    plt.subplot(235), plt.axis("off"), plt.title("(5) Final marker")
    plt.imshow(Fnew, cmap='gray')  # 收敛标记图像
    plt.subplot(236), plt.axis("off"), plt.title("(6) Rebuild image")
    plt.imshow(cv.bitwise_not(imgRebuild), cmap='gray')
    plt.tight_layout()
    plt.show()
