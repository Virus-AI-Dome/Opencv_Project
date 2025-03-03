"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【1808】基于BP算法多层神经网络的手写数字识别
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
    hog = cv.HOGDescriptor(_winSize=(20, 20),  # 检测窗口大小
                           _blockSize=(10, 10),  # 子块的大小
                           _blockStride=(10, 10),  # 子块的滑动步长->不滑动
                           _cellSize=(5, 5),  # 单元格大小
                           _nbins=8,  # 直方图的条数
                           _gammaCorrection=True)  # 伽马校正预处理
    descriptors = np.array([hog.compute(imgSamples[i]) for i in range(imgSamples.shape[0])])  # (5000, 128)
    samples = descriptors.astype(np.float32)  # 形状 (5000,128)
    m = np.arange(10)  # 输出值/分类标签 0~9，(10,)
    labels = np.tile(m, 500)[:, np.newaxis]  # 形状 (5000,1):012..9...012..9
    print(imgSamples.shape, descriptors.shape, samples.shape, labels.shape)

    # (3) 构造图像样本的分类标签和 one-hot 输出矩阵
    nSamples = samples.shape[0]
    nClasses = 10
    classes = np.arange(nClasses)  # 输出值/分类标签 0~9，(10,)
    category = np.tile(classes, 500).reshape(-1, 1)  # 形状 (5000,1):012..9...012..9
    # 分类标签 one-hot 编码，(5000, 1)->(5000, 10)，float32
    labelsMat = np.zeros((nSamples, nClasses), np.float32)  # 输出矩阵 (nSamples, nClasses)
    for j in range(nSamples):
        labelsMat[j, category[j]] = 1.0  # one-hot 编码
    # for k in classes:  # 效率更高
    #     labelsMat[np.where(category[:,0]==k), k] = 1.0  # one-hot 编码
    print(imgSamples.shape, descriptors.shape, samples.shape, labelsMat.shape)

    # (4) 构造训练样本集和检验样本集
    # 构造训练样本集的输入值 samplesTrain、输出值 labelsTrain
    samplesTrain = samples[0:4000]  # 形状 (4000,128)
    labelsTrain = labelsMat[0:4000]  # 形状 (4000,10)
    # trainData = cv.ml.TrainData_create(samplesTrain, cv.ml.ROW_SAMPLE, labelsTrain)  # 创建为 TrainData类
    categoryTrain = category[0:4000].astype(np.float32)  # 形状 (1000,1):012..9，float32
    print("TrainSamples:", samplesTrain.shape, labelsTrain.shape, categoryTrain.shape)
    # 构造测试样本集的输入值 samplesTest、输出值 labelsTest
    samplesTest = samples[4000:]  # 形状 (1000,128)
    categoryTest = category[4000:]  # 形状 (1000,1): 012..9
    print("TestnSamples:", samplesTest.shape, categoryTest.shape)

    # (5) ANN1 回归模型：1 个输出变量，浮点型
    # 创建和配置 ANN1 回归模型
    nInputs, nHiddennodes, nOutput = samples.shape[1], 128, categoryTrain.shape[1]  # 128, 128, 1
    ann1 = cv.ml.ANN_MLP_create()  # 创建 ANN 模型
    ann1.setLayerSizes(np.array([nInputs, nHiddennodes, nOutput]))  # 设置模型规模
    ann1.setActivationFunction(cv.ml.ANN_MLP_SIGMOID_SYM, 0.6, 1.0)  # 设置激活函数
    ann1.setTrainMethod(cv.ml.ANN_MLP_BACKPROP, 0.1, 0.1)  # 设置训练方法化参数
    ann1.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER | cv.TERM_CRITERIA_EPS, 1000, 0.001))  # 设置收敛准则
    # 训练 ANN1 模型
    # ann.train(trainData, cv.ml.ANN_MLP_NO_INPUT_SCALE|cv.ml.ANN_MLP_NO_OUTPUT_SCALE)  # 训练 ANN 模型
    ann1.train(samplesTrain, cv.ml.ROW_SAMPLE, categoryTrain)  # 训练 ANN 模型
    # 基于 ANN1 模型预测
    # nTrain = samplesTrain.shape[0]
    # _, result = ann1.predict(samplesTrain)  # 模型预测样本的分类结果，(1000,1)，float32
    # matches = (np.abs(result-categoryTrain)<0.5)  # 比较模型预测结果与样本标签
    # correct = np.count_nonzero(matches)  # 预测正确的样本数量
    # accuracy = 100 * correct / nTrain  # 模型预测的准确率
    # print("(1) ANN model with single output:")
    # print("\tnInputs={}, nHiddennodes={}, nOutput={}".format(nInputs, nHiddennodes, nOutput))
    # print("\tsamples={}, correct={}, accuracy={:.2f}%".format(nTrain, correct, accuracy))
    nTest = samplesTest.shape[0]
    _, result = ann1.predict(samplesTest)  # 模型预测样本的分类结果，(1000,1)，float32
    matches = (np.abs(result-categoryTest)<0.5)  # 比较模型预测结果与样本标签
    correct = np.count_nonzero(matches)  # 预测正确的样本数量
    accuracy = 100 * correct / nTest  # 模型预测的准确率
    print("(1) ANN1 regression model with single output:")
    print("\tnInputs={}, nHiddennodes={}, nOutput={}".format(nInputs, nHiddennodes, nOutput))
    print("\tsamples={}, correct={}, accuracy={:.2f}%".format(nTest, correct, accuracy))

    # (6) ANN2 分类模型：10 个输出变量，二分类
    # 创建和配置 ANN2 分类模型
    nInputs, nHiddennodes, nOutput = samples.shape[1], 64, labelsMat.shape[1]  # 128, 64, 10
    ann2 = cv.ml.ANN_MLP_create()  # 创建 ANN 模型
    ann2.setLayerSizes(np.array([nInputs, nHiddennodes, nOutput]))  # 设置模型规模
    ann2.setActivationFunction(cv.ml.ANN_MLP_SIGMOID_SYM, 0.6, 1.0)  # 设置激活函数
    ann2.setTrainMethod(cv.ml.ANN_MLP_BACKPROP, 0.1, 0.1)  # 设置训练方法和参数
    ann2.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER | cv.TERM_CRITERIA_EPS, 1000, 0.01))  # 设置收敛准则
    # 训练 ANN2 模型
    # ann2.train(trainData, cv.ml.ANN_MLP_NO_INPUT_SCALE|cv.ml.ANN_MLP_NO_OUTPUT_SCALE)  # 训练 ANN 模型
    ann2.train(samplesTrain, cv.ml.ROW_SAMPLE, labelsTrain)  # 训练 ANN 模型
    # 基于 ANN2 模型预测
    # nTrain = samplesTrain.shape[0]
    # _, result = ann2.predict(samplesTrain)  # 模型预测样本的分类结果，(4000,10)
    # predTrain = result.argmax(axis=1).reshape((-1, 1))  # 将分类结果转换为类别序号 (1000,1)
    # matches = (predTrain==categoryTrain)  # 比较模型预测结果与样本标签
    # correct = np.count_nonzero(matches)  # 预测正确的样本数量
    # accuracy = 100 * correct / nTrain  # 模型预测的准确率
    # print("(2) ANN classification model with multi outputs:")
    # print("\tnInputs={}, nHiddennodes={}, nOutput={}".format(nInputs, nHiddennodes, nOutput))
    # print("\tsamples={}, correct={}, accuracy={:.2f}%".format(nTrain, correct, accuracy))
    nTest = samplesTest.shape[0]
    _, result = ann2.predict(samplesTest)  # 模型预测样本的分类结果，(1000,10)
    predTest = result.argmax(axis=1).reshape((-1, 1))  # 将分类结果转换为类别序号 (1000,1)Tool
    matches = (predTest==categoryTest)  # 比较模型预测结果与样本标签
    correct = np.count_nonzero(matches)  # 预测正确的样本数量
    accuracy = 100 * correct / nTest  # 模型预测的准确率
    print("(2) ANN2 classification model with multi outputs:")
    print("\tnInputs={}, nHiddennodes={}, nOutput={}".format(nInputs, nHiddennodes, nOutput))
    print("\tsamples={}, correct={}, accuracy={:.2f}%".format(nTest, correct, accuracy))
