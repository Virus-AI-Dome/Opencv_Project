"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【1213】形态学重建之孔洞填充
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv.imread("../images/Fig1205.png", flags=0)  # 灰度图像
    _, imgBin = cv.threshold(img, 205, 255, cv.THRESH_BINARY_INV)  # 二值处理 (黑色背景)
    imgBinInv = cv.bitwise_not(imgBin)  # 二值图像的补集 (白色背景)，用于构造标记图像

    # 构造标记图像:
    F0 = np.zeros(imgBinInv.shape, np.uint8)  # 边界为 imgBinInv，其它全黑
    F0[:, 0] = imgBinInv[:, 0]
    F0[:, -1] = imgBinInv[:, -1]
    F0[0, :] = imgBinInv[0, :]
    F0[-1, :] = imgBinInv[-1, :]

    # 形态学重建
    Flast = F0.copy()  # F(k)
    element = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    iter= 0
    while True:
        dilateF = cv.dilate(Flast, kernel=element)  # 标记图像膨胀
        Fnew = cv.bitwise_and(dilateF, imgBinInv)  # 原图像的补集作为模板约束重建
        if (Fnew==Flast).all():  # 收敛判断，F(k+1)=F(k)?
            break  # 结束迭代，Fnew 是收敛的标记图像
        else:
            Flast = Fnew.copy()  # 更新 F(k)
        iter += 1  # 迭代次数
        if iter==2: imgF1 = Fnew  # 显示中间结果
        elif iter==100: imgF100 = Fnew  # 显示中间结果
    print("iter=", iter)
    imgRebuild = cv.bitwise_not(Fnew)  # F(k) 的补集是孔洞填充的重建结果

    plt.figure(figsize=(9, 5.6))
    plt.subplot(231), plt.axis("off"), plt.title("(1) Original")
    plt.imshow(img, cmap='gray')
    plt.subplot(232), plt.axis("off"), plt.title("(2) Template")
    plt.imshow(imgBinInv, cmap='gray')  # 白色背景
    plt.subplot(233), plt.axis("off"), plt.title("(3) Initial marker")
    plt.imshow(imgF1, cmap='gray')
    plt.subplot(234), plt.axis("off"), plt.title("(4) Marker (iter=100)")
    plt.imshow(imgF100, cmap='gray')
    plt.subplot(235), plt.axis("off"), plt.title("(5) Final Marker")
    plt.imshow(Fnew, cmap='gray')
    plt.subplot(236), plt.axis("off"), plt.title("(6) Rebuild image")
    plt.imshow(imgRebuild, cmap='gray')
    plt.tight_layout()
    plt.show()
