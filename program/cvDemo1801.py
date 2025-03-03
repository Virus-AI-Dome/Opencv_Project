"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【1801】基于主成分分析的特征提取与图像重建
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    # 读取光谱图像组
    img = cv.imread("../images/Fig1801a.tif", flags=0)
    height, width = img.shape[:2]  # (564, 564)
    nBands = 6  # 光谱波段种类
    snBands = ['a', 'b', 'c', 'd', 'e', 'f']  # Fig1138a~f
    imgMulti = np.zeros((height, width, nBands))  # (564,564,6)
    Xmat = np.zeros((img.size, nBands))  # (318096, 6)
    print(imgMulti.shape, Xmat.shape)
    # 显示光谱图像组
    fig1 = plt.figure(figsize=(9, 6))  # 原始图像，6 个不同波段
    fig1.suptitle("Spectral image of multi bands by NASA")
    for i in range(nBands):
        path = "../images/Fig1801{}.tif".format(snBands[i])
        imgMulti[:, :, i] = cv.imread(path, flags=0)
        ax1 = fig1.add_subplot(2,3,i+1)
        ax1.set_xticks([]), ax1.set_yticks([])
        ax1.imshow(imgMulti[:,:,i], 'gray')  # 绘制光谱图像 snBands[i]
    plt.tight_layout()

    # 主成分分析 (principal component analysis)
    m, p = Xmat.shape  # m：训练集样本数量，p：特征维度数
    Xmat = np.reshape(imgMulti, (-1, nBands))  # (564,564,6) -> (318096,6)
    mean, eigVect, eigValue = cv.PCACompute2(Xmat, np.empty((0)), retainedVariance=0.98)
    #  mean, eigVect, eigValue = cv.PCACompute2(Xmat, np.empty((0)), maxComponents=3)
    print(mean.shape, eigVect.shape, eigValue.shape)  # (1, 6) (3, 6) (3, 1)
    eigenvalues = np.squeeze(eigValue)  # 删除维度为1的数组维度，(3,1)->(3,)

    # 保留的主成分数量
    K = eigVect.shape[0]  # 主成分方差贡献率 98% 时的特征维数 K=3
    print("number of samples: m=", m)  # 样本集的样本数量 m=318096
    print("number of features: p=", p)  # 样本集的特征维数 p=6
    print("number of PCA features: k=", K)  # 降维后的特征维数，主成分个数 k=3
    print("mean:", mean.round(4))  # 均值
    print("topK eigenvalues:\n", eigenvalues.round(4))  # 特征值，从大到小
    print("topK eigenvectors:\n", eigVect.round(4))  # (3, 6)

    # 压缩图像特征，将输入数据按主成分特征向量投影到 PCA 特征空间
    mbMatPCA = cv.PCAProject(Xmat, mean, eigVect)  # (318096, 6)->(318096, K=3)
    # 显示主成分变换图像
    fig2 = plt.figure(figsize=(9, 3.2))  # 主元素图像
    fig2.suptitle("Images of principal components")
    for i in range(K):
        pca = mbMatPCA[:, i].reshape(-1, img.shape[1])  # 主元素图像 (564, 564)
        imgPCA = cv.normalize(pca, (height, width), 0, 255, cv.NORM_MINMAX)
        ax2 = fig2.add_subplot(1,3,i+1)
        ax2.set_xticks([]), ax2.set_yticks([])
        ax2.imshow(imgPCA, 'gray')  # 绘制主成分图像
    plt.tight_layout()

    # # 由主成分分析重建图像
    reconMat = cv.PCABackProject(mbMatPCA, mean, eigVect)  # (318096, K=3)->(318096, 6)
    fig3 = plt.figure(figsize=(9, 6))  # 重建图像，6 个不同波段
    fig3.suptitle("Rebuild images from principal components")
    rebuild = np.zeros((height, width, nBands))  # (564, 564, 6)
    for i in range(nBands):
        rebuild = reconMat[:, i].reshape(-1, img.shape[1])  # 主元素图像 (564, 564)
        # rebuild = np.uint8(cv.normalize(rebuild, (height, width), 0, 255,  cv.NORM_MINMAX))
        ax3 = fig3.add_subplot(2,3,i+1)
        ax3.set_xticks([]), ax3.set_yticks([])
        ax3.imshow(rebuild, 'gray')  # 绘制光谱图像 snBands[i]
    plt.tight_layout()
    plt.show()



