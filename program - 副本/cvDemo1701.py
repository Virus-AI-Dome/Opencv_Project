"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【1701】角点检测之Harris算法和Shi-Tomas算法
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv.imread("../images/Fig1201.png", flags=1)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Harris 角点检测算法
    dst = cv.cornerHarris(gray, 5, 3, k=0.04)  # 角点响应图像，坐标(y,x)
    # Harris[dst>0.1*dst.max()] = [0,0,255]  # 筛选角点，红色标记
    stack = np.column_stack(np.where(dst>0.05*dst.max()))  # 阈值筛选角点 (n,2)
    corners = stack[:, [1, 0]]  # 调整坐标次序：(y,x) -> (x,y)
    print("num of corners by Harris: ", corners.shape)
    imgHarris = img.copy()
    for point in corners:
        cv.drawMarker(imgHarris, point, (0,0,255), cv.MARKER_CROSS, 10, 1)  # 在点(x,y)标记

    # Shi-Tomas 角点检测算法
    maxCorners, qualityLevel, minDistance = 100, 0.1, 5
    corners = cv.goodFeaturesToTrack(gray, maxCorners, qualityLevel, minDistance)  # 角点坐标 (x,y)
    corners = np.squeeze(corners).astype(np.int16)  # 检测到的角点 (n,1,2)->(n,2)
    print("num of corners by Shi-Tomas: ", corners.shape[0])
    imgShiTomas = np.copy(img)
    for point in corners:  # 注意坐标次序
        cv.drawMarker(imgShiTomas, (point[0], point[1]), (0,0,255), cv.MARKER_CROSS, 10, 2)  # 在点(x,y)标记

    plt.figure(figsize=(9, 3.3))
    plt.subplot(131), plt.title("(1) Original")
    plt.axis('off'), plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.subplot(132), plt.title("(2) Harris corners")
    plt.axis('off'), plt.imshow(cv.cvtColor(imgHarris, cv.COLOR_BGR2RGB))
    plt.subplot(133), plt.title("(3) Shi-tomas corners")
    plt.axis('off'), plt.imshow(cv.cvtColor(imgShiTomas, cv.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()