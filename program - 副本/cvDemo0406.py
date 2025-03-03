"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【0406】绘制多段线和多边形
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = np.ones((900, 400, 3), np.uint8) * 224
    img1 = img.copy()
    img2 = img.copy()
    img3 = img.copy()
    img4 = img.copy()

    # 多边形顶点
    points1 = np.array([[200, 60], [295, 129], [259, 241], [141, 241], [105, 129]])
    points2 = np.array([[200, 350], [259, 531], [105, 419], [295, 419], [141, 531]])  # (5,2)
    points3 = np.array([[200, 640], [222, 709], [295, 709], [236, 752], [259, 821],
                        [200, 778], [141, 821], [164, 752], [105, 709], [178, 709]])
    print(points1.shape, points2.shape, points3.shape)  # (5, 2) (5, 2) (10, 2)
    # 绘制多边形，闭合曲线
    pts1 = [points1]  # pts1 是列表，列表元素是形状为 (m,2) 的 numpy 二维数组
    cv.polylines(img1, pts1, True, (0, 0, 255))  # pts1  是列表
    cv.polylines(img1, [points2, points3], 1, 255, 2)  # 可以绘制多个多边形

    # 绘制多段线，曲线不闭合
    cv.polylines(img2, [points1], False, (0, 0, 255))
    cv.polylines(img2, [points2, points3], 0, 255, 2)  # 可以绘制多个多段线

    # 绘制填充多边形，注意交叉重叠部分处理
    cv.fillPoly(img3, [points1], (0, 0, 255))
    cv.fillPoly(img3, [points2, points3], 255)  # 可以绘制多个填充多边形

    # 绘制一个填充多边形，注意交叉重叠部分
    cv.fillConvexPoly(img4, points1, (0, 0, 255))
    cv.fillConvexPoly(img4, points2, 255)  # 不能绘制存在自相交的多边形
    cv.fillConvexPoly(img4, points3, 255)  # 可以绘制凹多边形，但要慎用

    plt.figure(figsize=(9, 5.3))
    plt.subplot(141), plt.title("(1) closed polygon"), plt.axis('off')
    plt.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
    plt.subplot(142), plt.title("(2) unclosed polygo"), plt.axis('off')
    plt.imshow(cv.cvtColor(img2, cv.COLOR_BGR2RGB))
    plt.subplot(143), plt.title("(3) fillPoly"), plt.axis('off')
    plt.imshow(cv.cvtColor(img3, cv.COLOR_BGR2RGB))
    plt.subplot(144), plt.title("(4) fillConvexPoly"), plt.axis('off')
    plt.imshow(cv.cvtColor(img4, cv.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()

