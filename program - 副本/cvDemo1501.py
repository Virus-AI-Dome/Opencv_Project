"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

#【1501】图像分割之区域生长
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# 区域生长算法
def regional_growth(img, seeds, thresh=5):
    height, width = img.shape
    seedMark = np.zeros(img.shape)  # (h,w)
    seedList = []  # (y,x)
    for seed in seeds:  # seeds (x,y)
        if (0<seed[0]<height and 0<seed[1]< width): seedList.append(seed)
    label = 1  # 种子位置标记
    connects = [(-1,-1), (0,-1), (1,-1), (1,0), (1,1), (0,1), (-1,1), (-1,0)]  # 8 邻域连通
    while (len(seedList) > 0):  # 如果列表里还存在点
        curPoint = seedList.pop(0)  # 抛出第0个
        seedMark[curPoint[0], curPoint[1]] = label  # 将对应位置标记为 1
        for i in range(8):  # 对 8 邻域点进行相似性判断
            tmpY = curPoint[0] + connects[i][0]
            tmpX = curPoint[1] + connects[i][1]
            if tmpY<0 or tmpX<0 or tmpY>=height or tmpX>=width:  # 是否超出限定阈值
                continue
            grayDiff = np.abs(int(img[curPoint[0], curPoint[1]]) - int(img[tmpY, tmpX]))  # 计算灰度差
            if grayDiff < thresh and seedMark[tmpY, tmpX] == 0:
                seedMark[tmpY, tmpX] = label
                seedList.append((tmpY, tmpX))
    imgGrowth = np.uint8(cv.normalize(seedMark, None, 0, 255, cv.NORM_MINMAX))
    return imgGrowth

if __name__ == '__main__':
    img = cv.imread("../images/Fig1501.png", flags=0)

    # OTSU 全局阈值处理，用于比较
    ret, imgOtsu = cv.threshold(img, 127, 255, cv.THRESH_OTSU)
    # 自适应局部阈值处理，用于比较
    binaryMean = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 5, 3)

    # 区域生长图像分割
    # seeds = [(10, 10), (82, 150), (20, 300)]  # 直接给定 种子点
    imgBlur = cv.blur(img, (3, 3))  # cv.blur 方法
    _, imgTop = cv.threshold(imgBlur, 205, 255, cv.THRESH_BINARY)  # 高百分位阈值产生种子区域
    nseeds, labels, stats, centroids = cv.connectedComponentsWithStats(imgTop)  # 过滤连通域，获得质心点 (x,y)
    seeds = centroids.astype(int)  # 获得质心像素作为种子点
    imgGrowth = regional_growth(img, seeds, 5)
    print(imgGrowth.max(), imgGrowth.min(), imgGrowth.mean())

    plt.figure(figsize=(9, 5.6))
    plt.subplot(231), plt.axis('off'), plt.title("(1) Original")
    plt.imshow(img, 'gray')
    plt.subplot(232), plt.axis('off'), plt.title("(2) OTSU(T={})".format(ret))
    plt.imshow(imgOtsu, 'gray')
    plt.subplot(233), plt.axis('off'), plt.title("(3) Adaptive threshold")
    plt.imshow(binaryMean, 'gray')
    plt.subplot(234, yticks=[])
    histSrc = cv.calcHist([img], [0], None, [256], [0, 255])
    plt.axis([0, 255, 0, np.max(histSrc)]), plt.title("(4) GrayHist of src")
    plt.bar(range(256), histSrc[:, 0])  # 原始图像直方图
    plt.subplot(235), plt.axis('off'), plt.title("(5) Marked seeds")
    plt.imshow(labels, 'gray', vmin=0, vmax=1)
    plt.subplot(236), plt.axis('off'), plt.title("(6) Region growth")
    invGrowth = cv.bitwise_not(imgGrowth)
    plt.imshow(invGrowth, 'gray')
    plt.tight_layout()
    plt.show()
