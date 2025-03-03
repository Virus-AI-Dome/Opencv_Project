"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【1604】特征描述之区域特征描述（紧致度、圆度和偏心率）
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv.imread("../images/Fig1603.png", flags=1)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 灰度图像
    print("shape of image:", gray.shape)

    # 查找图形轮廓
    _, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)  # OpenCV4~
    cnts = sorted(contours, key=cv.contourArea, reverse=True)  # 所有轮廓按面积排序
    cnt = cnts[0]  # 第 0 个轮廓，面积最大的轮廓，(2867, 1, 2)
    # cntPoints = np.squeeze(cnt)  # 删除维度为 1 的数组维度，(2867, 1, 2)->(2867,2)

    print("| :--: | :--: | :--: | :--: | :--: | :--: |")
    print("| 图形编号 | 面积 | 周长 | 紧致度 | 圆度 | 偏心率 |")
    print("| :--: | :--: | :--: | :--: | :--: | :--: |")
    contourEx = gray.copy()  # OpenCV3.2 之前的早期版本，查找轮廓函数会修改原始图像
    for i in range(len(cnts)):  # 绘制第 i 个轮廓
        x, y, w, h = cv.boundingRect(cnts[i])  # 外接矩形
        contourEx = cv.drawContours(contourEx, cnts, i, (0,0,255), thickness=5)  # 第 i 个轮廓，内部填充
        contourEx = cv.putText(contourEx, str(i)+"#", (x,y-20), cv.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0))

        area = cv.contourArea(cnts[i])  # 轮廓面积 (area)
        perimeter = cv.arcLength(cnts[i], True)  # 轮廓周长 (perimeter)
        compact = perimeter ** 2 / area  # 轮廓的紧致度 (compactness)
        circular = 4 * np.pi * area / perimeter ** 2  # 轮廓的圆度 (circularity)
        ellipse = cv.fitEllipse(cnts[i])  # 轮廓的拟合椭圆
        # 椭圆中心点 (x,y), 长轴短轴长度 (a,b), 旋转角度 ang
        (x, y), (a, b), ang = np.int32(ellipse[0]), np.int32(ellipse[1]), round(ellipse[2], 1)
        # 轮廓的偏心率 (eccentricity)
        if (a > b):
            eccentric = np.sqrt(1.0 - (b/a)**2)  # a 为长轴
        else:
            eccentric = np.sqrt(1.0 - (a/b)**2)  # b 为长轴

        print("| {} | {:.1f} | {:.1f} | {:.1f} | {:.1f} |{:.1f} |"
              .format(i+1, area,perimeter,compact,circular,eccentric))
    print("| :--: | :--: | :--: | :--: | :--: | :--: |")

    plt.figure(figsize=(8.5, 3.3))
    plt.subplot(121), plt.axis('off'), plt.title("(1) Original")
    plt.imshow(img, cmap='gray')
    plt.subplot(122), plt.axis('off'), plt.title("(2) Contours")
    plt.imshow(cv.cvtColor(contourEx, cv.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()
