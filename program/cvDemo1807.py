"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【1807】基于支持向量机的手写数字识别
import cv2 as cv
import numpy as np

if __name__ == '__main__':
    # (1) 读取样本图像，构造样本图像集合
    img = cv.imread("../images/digits.png")  # 5000 个手写数字，每个数字 20x20
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 将原始图像分割为 100×50=5000 个单元格，每个单元格是一个手写数字
    cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]  # (50,100)
    x1 = np.array(cells)  # 转换为 Numpy 数组，形状为 (50,100,20,20)
    x2 = np.reshape(x1, (-1, 20, 20))  # (5000,20,20): 00..011..1...99..9
    x3 = np.reshape(x2, (500, 10, 20, 20), order='F')  # (500,10,20,20)
    imgSamples = np.reshape(x3, (-1, 20, 20))  # 形状 (5000,20,20):012..9...012..9
    print(x1.shape, x2.shape, x3.shape, imgSamples.shape)

    # (2) 构造 HOG 描述符
    winSize = (20, 20)  # 检测窗口大小
    blockSize = (10, 10)  # 子块的大小
    blockStride = (10, 10)  # 子块的滑动步长->不滑动
    cellSize = (5, 5)  # 单元格大小
    nbins = 8  # 直方图的条数
    lenHOG = nbins * (blockSize[0]/cellSize[0]) * (blockSize[1]/cellSize[1]) \
            * ((winSize[0]-blockSize[0])/blockStride[0] + 1) \
            * ((winSize[1]-blockSize[1])/blockStride[1] + 1)
    hog = cv.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    descriptors = np.array([hog.compute(imgSamples[i]) for i in range(imgSamples.shape[0])])  # (5000, 128)
    samples = descriptors.astype(np.float32)  # 形状 (5000,128)
    m = np.arange(10)  # 输出值/分类标签 0~9，(10,)
    labels = np.tile(m, 500)[:, np.newaxis]  # 形状 (5000,1):012..9...012..9
    print(imgSamples.shape, descriptors.shape, samples.shape, labels.shape)

    # (3) 构造训练样本数据集和测试样本数据集
    # 训练样本集的输入值 samplesTrain、输出值 labelsTrain
    samplesTrainH = samples[0:4000]  # 形状 (4000,128)
    labelsTrainH = labels[0:4000]  # 形状 (4000,1)
    print(samplesTrainH.shape, labelsTrainH.shape)
    # 测试样本集的输入值 samplesTest、输出值 labelsTest
    samplesTestH = samples[4000:]  # 形状 (1000,128)
    labelsTestH = labels[4000:]  # 形状 (1000,1)
    print(samplesTestH.shape, labelsTestH.shape)

    # (4) SVM 模型的创建、训练和预测
    svm = cv.ml.SVM_create()  # 创建 SVM 模型
    svm.setType(cv.ml.SVM_C_SVC)  # 设置模型类型为 分类问题，软间隔
    svm.setKernel(cv.ml.SVM_RBF)  # 设置核函数类型为 RBF 核
    svm.train(samplesTrainH, cv.ml.ROW_SAMPLE, labelsTrainH)  # 训练 SVM 模型
    _, resultH = svm.predict(samplesTestH)  # SVM 模型预测，使用模型对样本进行分类

    matches = (resultH==labelsTestH)  # 比较模型预测结果与样本标签
    correct = np.count_nonzero(matches)  # 预测正确的样本数量
    accuracy = correct * 100.0 / resultH.size  # 模型预测的准确率
    print("features={}, correct={}, accuracy={:.2f}%".format(lenHOG, correct, accuracy))
