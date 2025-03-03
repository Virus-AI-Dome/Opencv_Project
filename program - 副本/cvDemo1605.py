"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【1605】特征描述之灰度共生矩阵
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def getGlcm(src, dx, dy, grayLevel=16):  # 计算灰度共生矩阵 GLCM
    height, width = src.shape[:2]
    grayLevel = src.max() + 1
    glcm = np.zeros((grayLevel, grayLevel), np.int)  # (16, 16)
    for j in range(height - dy):
        for i in range(width - dx):
            rows = src[j][i]
            cols = src[j + dy][i + dx]
            glcm[rows][cols] += 1
    return glcm / glcm.max()  # -> (0.0,1.0)

def calGlcmProps(glcm, grayLevel=16):
    Asm, Con, Ent, Idm = 0.0, 0.0, 0.0, 0.0
    for i in range(grayLevel):
        for j in range(grayLevel):
            Con += (i - j) * (i - j) * glcm[i][j]  # 对比度
            Asm += glcm[i][j] * glcm[i][j]  # 能量
            Idm += glcm[i][j] / (1 + (i - j) * (i - j))  # 反差分矩阵
            if glcm[i][j] > 0.0:
                Ent += glcm[i][j] * np.log(glcm[i][j])
    return Asm, Con, -Ent, Idm

if __name__ == '__main__':
    from skimage.feature import greycomatrix, greycoprops
    img = cv.imread("../images/Fig1604.png", flags=1)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 灰度图像
    height, width = gray.shape

    # 将灰度级压缩到 16 级
    table16 = np.array([(i//16) for i in range(256)]).astype(np.uint8)  # 16 levels
    gray16 = cv.LUT(gray, table16)  # 灰度级压缩为 [0,15]

    # 计算灰度共生矩阵 GLCM
    dist = [1, 4]  # 计算 2 个距离偏移量 [1, 2]
    degree = [0, np.pi/4, np.pi/2, np.pi*3/4]  # 计算 4 个方向
    glcm = greycomatrix(gray16, dist, degree, levels=16)  # 灰度级 L=16
    print("glcm.shape:", glcm.shape)  # (16,16,2,4)

    # 由灰度共生矩阵 GLCM 计算特征统计量
    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
        feature = greycoprops(glcm, prop).round(2)  # (2,4)
        print("{}: {}".format(prop, feature))

    plt.figure(figsize=(9, 5.5))
    plt.suptitle("GLCM by skimage")
    for i in range(len(dist)):
        for j in range(len(degree)):
            plt.subplot(2, 4, i*4+j+1), plt.axis('off')
            plt.title(r"d={},$\theta$={:.2f}".format(dist[i], degree[j]))
            plt.imshow(glcm[:, :, i, j], 'gray')
    plt.tight_layout()
    plt.show()
