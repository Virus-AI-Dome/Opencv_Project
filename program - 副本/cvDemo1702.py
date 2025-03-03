"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【1702】角点检测之亚像素精确定位
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv.imread("../images/Fig1209.png", flags=1)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # (400, 400)
    print(gray.shape)  # (400, 400)

    # Shi-Tomas 检测角点
    maxCorners, qualityLevel, minDistance = 100, 0.2, 5
    corners = cv.goodFeaturesToTrack(gray, maxCorners, qualityLevel, minDistance)  # 角点坐标 (x,y)
    corners = np.squeeze(corners).astype(np.int)  # 检测到的角点 (n,1,2)->(n,2)
    print("num of corners by Shi-Tomas: ", corners.shape[0])
    imgShiTomas = img.copy()
    for point in corners:
        cv.drawMarker(imgShiTomas, point, (255,0,0), cv.MARKER_CROSS, 10, 1)  # 在点(x,y)标记

    # # 对角点进行精细定位
    winSize = (3, 3)  # 搜索窗口的半尺寸
    zeroZone = (-1, -1)  # 搜索盲区的半尺寸
    criteria = (cv.TERM_CRITERIA_EPS+cv.TERM_CRITERIA_MAX_ITER, 50, 0.01)  # 终止判据
    fineCorners = cv.cornerSubPix(gray, np.float32(corners), winSize, zeroZone, criteria)
    print("shape of fineCorners: ", fineCorners.shape)
    for i in range(corners.shape[0]):
        if np.max(np.abs(corners[i]-fineCorners[i]))>1:
            xp, yp = fineCorners[i,0], fineCorners[i,1]
            print("corners={}, subPix=[{:.1f},{:.1f}]".format(corners[i], xp, yp))

    # 精细定位检测图像
    fineCorners = fineCorners.astype(np.int32)  # 精细角点坐标 (x,y)
    imgSubPix = img.copy()
    for point in fineCorners:
        cv.drawMarker(imgSubPix, point, (0,0,255), cv.MARKER_CROSS, 10, 1)  # 在点(x,y)标记

    plt.figure(figsize=(9, 3.5))
    plt.subplot(131), plt.axis('off'), plt.title("(1) Shi-Tomas corners")
    plt.imshow(cv.cvtColor(imgShiTomas, cv.COLOR_BGR2RGB))
    plt.subplot(132), plt.axis('off'), plt.title("(2) Partial enlarge")
    plt.imshow(cv.cvtColor(imgShiTomas[:100,50:150], cv.COLOR_BGR2RGB))
    plt.subplot(133), plt.axis('off'), plt.title("(3) SubPix corners")
    plt.imshow(cv.cvtColor(imgSubPix[:100,50:150], cv.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()
