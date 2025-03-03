"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【1411】基于Hu不变矩的形状相似性检测
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv.imread("../images/Fig1404.png", flags=1)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 灰度图像
    _, binary = cv.threshold(gray, 127, 255, cv.THRESH_OTSU + cv.THRESH_BINARY_INV)

    # 寻找二值化图中的轮廓
    # binary, contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)  # OpenCV3
    contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # OpenCV4~
    #  绘制最外层轮廓
    contourEx = img.copy()  # OpenCV3.2 之前的早期版本，查找轮廓函数会修改原始图像
    for i in range(len(contours)):  # 绘制第 i 个轮廓
        x, y, w, h = cv.boundingRect(contours[i])  # 外接矩形
        contourEx = cv.drawContours(contourEx, contours, i, (205, 0, 0), thickness=-1)  # 第 i 个轮廓，内部填充
        contourEx = cv.putText(contourEx, str(i)+"#", (x,y-10), cv.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0))

    # 形状相似度检测
    print("| 对比形状 | 相似度 | 变形方式 |")
    print("| :--: | :--: | :--: |")
    similarity = np.array([cv.matchShapes(contours[7], contours[i], 1, 0.0) for i in range(len(contours))])
    argSort = similarity.argsort()  # 形状相似度 ret 从小到大排序 (相似度降低)
    for i in range(len(contours)):
        index = argSort[i]
        print("| cnt[7] & cnt[{}] | {} |  |".format(index, round(similarity[index], 2)))

    # 计算所有轮廓的 Hu 不变矩
    print("| 轮廓编号 | M1 | M2 | M3 | M4 | M5 | M6 | M7 |")
    print("| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |")
    cntsHuM = np.empty((len(contours), 7), np.float32)
    for i in range(len(contours)):
        moments = cv.moments(contours[i])  # 返回字典，几何矩, 中心矩和归一化矩
        hum = cv.HuMoments(moments)  # 计算 Hu 不变矩
        cntsHuM[i, :] = np.round(hum.reshape(hum.shape[0]), 2)
        print("|{}|{:.2e}|{:.2e}|{:.2e}|{:.2e}|{:.2e}|{:.2e}|{:.2e}|"
              .format(i, cntsHuM[i][0], cntsHuM[i][1], cntsHuM[i][2],
               cntsHuM[i][3], cntsHuM[i][4], cntsHuM[i][5], cntsHuM[i][6]))

    plt.figure(figsize=(8.5, 3.2))
    plt.subplot(121), plt.axis('off'), plt.title("(1) Original")
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.subplot(122), plt.axis('off'), plt.title("(2) Contours")
    plt.imshow(cv.cvtColor(contourEx, cv.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()
