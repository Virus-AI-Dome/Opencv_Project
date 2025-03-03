"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

#【1802】基于 k 均值聚类的图像减色处理
import cv2 as cv
import numpy as np
from matplotlib import cm, pyplot as plt

if __name__ == '__main__':
    # (1) 读取图像，构造样本数据矩阵 (h*w, 3)
    img = cv.imread("../images/Fig1701.png", flags=1)
    dataPixel = np.float32(img.reshape((-1, 3)))  # (250000, 3)
    print("dataPixel:", dataPixel.shape)

    # (2) k-Means 参数设置
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 200, 0.1)  # 终止条件
    flags = cv.KMEANS_RANDOM_CENTERS  # 起始的中心选择

    # (3) k-Means 聚类
    K = 2  # 设置聚类数量
    compactness, labels, center = cv.kmeans(dataPixel, K, None, criteria, 10, flags)
    centerUint = np.uint8(center)  # (K,3)
    classify = centerUint[labels.flatten()]  # 将像素标记为聚类中心颜色，(h*w,3)
    imgKmean1 = classify.reshape((img.shape))  # 恢复为二维图像，(h,w,3)
    labels2D1 = labels.reshape((img.shape[:2])) * 255/K  # 恢复为二维分类
    print("K=2, center:", center.shape)  # (k, 2)

    K = 3  # 设置聚类数量
    _, labels, center = cv.kmeans(dataPixel, K, None, criteria, 10, flags)
    centerUint = np.uint8(center)
    classify = centerUint[labels.flatten()]  # 将像素标记为聚类中心颜色
    imgKmean2 = classify.reshape((img.shape))  # 恢复为二维图像
    labels2D2 = labels.reshape((img.shape[:2])) * 255/K  # 恢复为二维分类
    print("K=3, center:", center.shape)  # (k, 3)

    K = 4  # 设置聚类数量
    _, labels, center = cv.kmeans(dataPixel, K, None, criteria, 10, flags)
    centerUint = np.uint8(center)
    classify = centerUint[labels.flatten()]  # 将像素标记为聚类中心颜色
    imgKmean3 = classify.reshape((img.shape))  # 恢复为二维图像
    labels2D3 = labels.reshape((img.shape[:2])) * 255/K  # 恢复为二维分类
    print("K=4, center:", center.shape)  # (k, 4)

    plt.figure(figsize=(9, 6.2))
    plt.subplot(231), plt.axis('off'), plt.title("(1) k-means (k=2)")
    plt.imshow(cv.cvtColor(imgKmean1, cv.COLOR_BGR2RGB))
    plt.subplot(232), plt.axis('off'), plt.title("(2) k-means (k=3)")
    plt.imshow(cv.cvtColor(imgKmean2, cv.COLOR_BGR2RGB))
    plt.subplot(233), plt.axis('off'), plt.title("(3) k-means (k=4)")
    plt.imshow(cv.cvtColor(imgKmean3, cv.COLOR_BGR2RGB))
    plt.subplot(234), plt.axis('off'), plt.title("(4) Tagging (k=2)")
    plt.imshow(labels2D1, 'gray')
    plt.subplot(235), plt.axis('off'), plt.title("(5) Tagging (k=3)")
    plt.imshow(labels2D2, 'gray')
    plt.subplot(236), plt.axis('off'), plt.title("(6) Tagging (k=4)")
    plt.imshow(labels2D3, 'gray')
    plt.tight_layout()
    plt.show()