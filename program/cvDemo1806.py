"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【1806】基于支持向量机的数据分类
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets, model_selection, metrics

if __name__ == '__main__':
    # 生成样本数据集，划分为训练样本集和检验样本集
    X, y = datasets.make_moons(n_samples=120, noise=0.1, random_state=306)  # 生成数据集
    xFloat = X.astype(np.float32)
    xTrain, xTest, yTrain, yTest = model_selection.train_test_split(xFloat, y, test_size=0.25, random_state=36)
    print(xTrain.shape, yTrain.shape, xTest.shape, yTest.shape)  # (75, 2) (75,) (25, 2) (25,)

    plt.figure(figsize=(8, 6))
    kernels = ['SVM_LINEAR', 'SVM_POLY', 'SVM_SIGMOID', 'SVM_RBF']  # 核函数的类型
    for idx, kernel in enumerate(kernels):
        svm = cv.ml.SVM_create()  # 创建 SVM 模型
        svm.setKernel(eval('cv.ml.' + kernel))  # 设置核函数类型
        if kernel=="SVM_POLY": svm.setDegree(3)  # POLY 阶数
        svm.train(xTrain, cv.ml.ROW_SAMPLE, yTrain)  # 用训练样本 (xTrain,yTrain) 训练模型
        _, yPred = svm.predict(xTest)  # 模型预测 检验样本 xText 的输出
        accuracy = metrics.accuracy_score(yTest, yPred) * 100  # 计算预测准确率

        # 计算分隔边界
        hmin, hmax = xTest[:,0].min(), xTest[:,0].max()
        vmin, vmax = xTest[:,1].min(), xTest[:,1].max()
        h = np.linspace(hmin, hmax, 100).astype(np.float32)  # (100,)
        v = np.linspace(vmin, vmax, 100).astype(np.float32)  # (100,)
        hGrid, vGrid = np.meshgrid(h, v)  # 生成网格点坐标矩阵 (100, 100)
        hvRavel = np.vstack([hGrid.ravel(), vGrid.ravel()]).T  # 将网格矩阵展平后重构为数组 (m,2)
        _, z = svm.predict(hvRavel)  # 模型估计 网格点的分类 (m,1)
        zGrid = z.reshape(hGrid.shape)  # 恢复为网格形状 (100, 100)

        # 绘图
        plt.subplot(2, 2, idx+1)
        plt.plot(xTest[:,0][yTest==0], xTest[:,1][yTest==0], "bs")  # 绘制第0类样本点
        plt.plot(xTest[:,0][yTest==1], xTest[:,1][yTest==1], "ro")  # 绘制第1类样本点
        plt.contourf(hGrid, vGrid, zGrid, cmap=plt.cm.brg, alpha=0.1)  # 绘制分隔平面
        plt.xticks([]), plt.yticks([])
        plt.title("({}) {} (accuracy={:.1f}%)".format(idx+1, kernel, accuracy))
        print("{} accuracy: {:.1f}%".format(kernel, accuracy))

    plt.tight_layout()
    plt.show()
