"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【1504】基于距离变换的分水岭算法
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv.imread("../images/Fig0301.png", flags=1)  # 彩色图像(BGR)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 灰度图像

    # 阈值分割，将灰度图像分为黑白二值图像
    ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_OTSU)
    # 形态学操作，生成确定背景区域 sureBG
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))  # 生成 3*3 结构元
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)  # 开运算，消除噪点
    sureBG = cv.dilate(opening, kernel, iterations=3)  # 膨胀操作，生成 "确定背景" 区域
    # 距离变换，生成确定前景区域 sureFG
    distance = cv.distanceTransform(opening, cv.DIST_L2, 5)  # DIST_L2: 3/5
    _, sureFG = cv.threshold(distance, 0.1 * distance.max(), 255, cv.THRESH_BINARY)  # 阈值选择 0.1*max 效果较好
    sureFG = np.uint8(sureFG)
    # 连通域处理
    ret, component = cv.connectedComponents(sureFG, connectivity=8)  # 对连通区域进行标号，序号为 0-N-1
    markers = component + 1  # OpenCV 分水岭算法设置标注从 1 开始，而连通域编从 0 开始
    kinds = markers.max()  # 标注连通域的数量
    maxKind = np.argmax(np.bincount(markers.flatten()))  # 出现最多的序号，所占面积最大，选为底色
    markersBGR = np.ones_like(img) * 255
    for i in range(kinds):
        if (i!=maxKind):
            colorKind = np.random.randint(0, 255, size=(1, 3))
            markersBGR[markers==i] = colorKind
    # 去除连通域中的背景区域部分
    unknown = cv.subtract(sureBG, sureFG)  # 待定区域，前景与背景的重合区域
    markers[unknown==255] = 0  # 去掉属于背景的区域 (置零)
    # 分水岭算法标注目标的轮廓
    markers = cv.watershed(img, markers)  # 分水岭算法，将所有轮廓的像素点标注为 -1

    # 把轮廓添加到原始图像上
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[markers==-1] = 255
    mask = cv.dilate(mask, kernel=np.ones((3,3)))  # 轮廓膨胀，使显示更明显
    imgWatershed = img.copy()
    imgWatershed[mask==255] = [255, 255, 255]  # 将分水岭算法标注的轮廓点设为蓝色

    plt.figure(figsize=(9, 5.8))
    plt.subplot(231), plt.axis('off'), plt.title("(1) Original")
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.subplot(232), plt.axis('off'), plt.title("(2) Gray image")
    plt.imshow(gray, 'gray')
    plt.subplot(233), plt.axis('off'), plt.title("(3) Sure background")
    plt.imshow(sureBG, 'gray')  # 确定背景
    plt.subplot(234), plt.axis('off'), plt.title("(4) Sure frontground")
    plt.imshow(sureFG, 'gray')  # 确定前景
    plt.subplot(235), plt.axis('off'), plt.title("(5) Markers")
    plt.imshow(cv.cvtColor(markersBGR, cv.COLOR_BGR2RGB))
    plt.subplot(236), plt.axis('off'), plt.title("(6) Watershed")
    plt.imshow(cv.cvtColor(imgWatershed, cv.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()
