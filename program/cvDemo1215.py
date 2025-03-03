"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【1215】基于重建开运算提取骨架
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv.imread("../images/Fig1206.png", flags=0)  # 读取为灰度图像
    _, imgBin = cv.threshold(img, 127, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)  # 二值处理

    element = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))  # 十字形结构元
    skeleton = np.zeros(imgBin.shape, np.uint8)  # 创建空骨架图
    Fk = imgBin.copy()  # 标记图像 F(k) 初值
    while True:
        imgOpen = cv.morphologyEx(Fk, cv.MORPH_OPEN, element)  # 开运算
        subSkel = cv.subtract(Fk, imgOpen)  # 获得骨架子集
        skeleton = cv.bitwise_or(skeleton, subSkel)  # 将删除的像素添加到骨架图
        if cv.countNonZero(Fk) == 0:  # 收敛判断 F(k)=0?
            break  # 结束迭代
        else:
            Fk = cv.erode(Fk, element)  # 更新 F(k)
    skeleton = cv.dilate(skeleton, element)  # 膨胀以方便显示，非必须步骤
    result = cv.bitwise_xor(img, skeleton)

    plt.figure(figsize=(9, 3.2))
    plt.subplot(131), plt.axis('off'), plt.title("(1) Original")
    plt.imshow(img, cmap='gray')
    plt.subplot(132), plt.axis('off'), plt.title("(2) Skeleton")
    plt.imshow(cv.bitwise_not(skeleton), cmap='gray')
    plt.subplot(133), plt.axis('off'), plt.title("(3) Stacked")
    plt.imshow(result, cmap='gray')
    plt.tight_layout()
    plt.show()


