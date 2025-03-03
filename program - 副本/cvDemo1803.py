"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

#【1803】基于KNN模型的手写数字识别
# 使用 OpenCV 自带的手写数字样本集，通过kNN方法进行手写数字识别
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    # (1) 读取样本图像，构造样本图像集合
    img = cv.imread("../images/digits.png")  # 5000 个手写数字，每个数字 20x20
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 将原始图像分割为 100×50=5000 个单元格，每个单元格是一个手写数字
    cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]  # (50,100)
    x = np.array(cells)  # 转换为 Numpy 数组，形状为 (50,100,20,20)
    # (2) 构造训练样本集，输入值 samplesTrain 输出值 labelsTrain
    samplesTrain = x[:, :80].reshape(-1, 400).astype(np.float32)  # (4000,400)
    m = np.arange(10)  # 输出值/分类标签 0~9，(10,)
    labelsTrain = np.repeat(m, 400)[:, np.newaxis]  # 样本标签 (4000,1)
    # trainData = cv.ml.TrainData_create(samplesTrain.astype(np.float32),
    #             cv.ml.ROW_SAMPLE, labelsTrain)  # 创建为 TrainData类，供参考
    # print(m.shape, samplesTrain.shape, labelsTrain.shape, type(trainData))
    # (3) 构造测试样本集，输入值 samplesTest 输出值 labelsTest
    samplesTest = x[:, 80:100].reshape(-1, 400).astype(np.float32)  # (1000,400)
    labelsTest = np.repeat(m, 100)[:, np.newaxis]  # 样本标签 (1000,1)
    print(m.shape, samplesTest.shape, labelsTest.shape)
    # # 将样本数据保存到文件
    # np.savez("knn_data.npz", samplesTrain=samplesTrain, labelsTrain=labelsTrain)
    # # 从文件中读取样本数据文件
    # with np.load("knn_data.npz") as data:
    #     print(data.files)
    #     samplesTrain = data["samplesTrain"]
    #     labelsTrain = data["labelsTrain"]

    # KNN 模型
    # (1) 创建 KNN 模型
    knn = cv.ml.KNearest_create()
    # (2) 训练 KNN 模型，samples 是输入向量，labels 是分类标记
    knn.train(samplesTrain, cv.ml.ROW_SAMPLE, labelsTrain)
    # knn.train(trainData)  # 用 trainData 训练 KNN 模型
    # (3) 模型检验，用训练好的模型进行分类，并与正确结果进行比较
    for k in range(2, 6):  # 不同 K 值的影响
        ret, result, neighbours, dist = knn.findNearest(samplesTest, k)  # 模型预测
        matches = (result==labelsTest)  # 模型预测结果与样本标签比较
        correct = np.count_nonzero(matches)  # 预测结果正确的样本数量
        accuracy = correct * 100.0 / result.size  # 模型预测的准确率
        print("k={}, correct={}, accuracy={:.2f}%".format(k, correct, accuracy))

