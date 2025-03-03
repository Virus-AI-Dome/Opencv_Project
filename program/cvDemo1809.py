"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【1809】基于多层神经网络的多光谱数据分类
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    # (1) 读取光谱图像组 Fig1138a~f 和 标记图像    nBands = 4  # 光谱波段种类
    nBands = 4  # 光谱波段种类
    height, width = (512, 512)
    snBands = ['a', 'b', 'c', 'd', 'e', 'f']  # Fig1138a~f
    imgMulti = np.zeros((height, width, nBands), np.uint8)  # (512,512,6)
    for i in range(nBands):
        path = "../images/Fig1801{}.tif".format(snBands[i])
        imgMulti[:,:,i] = cv.imread(path, flags=0)
    # 标记图像，1/2/3 表示 市区/植被/水体 区域，0 表示未知
    imgLabel = cv.imread("../images/Fig1801L.tif", flags=0)  # 市区/植被/水体

    # (2) 读取标记图像组 Fig1138Label1~3
    # imgLabel1 = cv.imread("../images/Fig1801Label1.tif", flags=0)  # 市区
    # imgLabel2 = cv.imread("../images/Fig1801Label2.tif", flags=0)  # 植被
    # imgLabel3 = cv.imread("../images/Fig1801Label3.tif", flags=0)  # 水体
    # imgLabel = np.zeros((height, width), np.uint8)

    # (2) 构造带标记的训练样本集
    # 生成 第 0 类 样本数据：市区
    Xmat0 = imgMulti[np.where(imgLabel==1)]  # 市区 (1865, nBands)
    num0 = Xmat0.shape[0]  # 1865
    label0 = np.zeros((num0,1), np.int16)  # 标记值 0，(1865, 1)
    # 生成 第 1 类 样本数据：植被
    Xmat1 = imgMulti[np.where(imgLabel==2)]  # 植被 (965, 4)
    num1 = Xmat1.shape[0]  # 965
    label1 = np.ones((num1,1), np.int16)  # 标记值 1，(965, 1)
    # 生成 第 2 类 样本数据：水体
    Xmat2 = imgMulti[np.where(imgLabel==3)]  # 水体 (967, 4)
    num2 = Xmat2.shape[0]  # 967
    label2 = np.ones((num2,1), np.int16)*2  # 标记值 2，(967, 1)
    print("num of label 1/2/3: {}/{}/{}".format(num0, num1, num2))

    # (3) 构造训练样本集和预测样本集的输入值 samples 和 One-hot 编码的输出值 labels
    # 拼接第 1/2/3 类样本数据，作为训练样本集
    samplesTrain = np.vstack((Xmat0, Xmat1, Xmat2)).astype(np.float32)  # (3797, 4)
    categoryTrain = np.vstack((label0, label1, label2))  # (3797, 1)
    # 构造训练样本集的分类标签 one-hot 编码，(5000, 1)->(5000, 3)，uint8
    nSamples, nClasses = samplesTrain.shape[0], 3  # 分为 3 类
    labelsMat = np.zeros((nSamples, nClasses), np.float32)  # 输出矩阵 (3797, nClasses=3)
    print(nSamples, nClasses, labelsMat.shape)
    for j in range(nSamples):
        labelsMat[j, categoryTrain[j]] = 1  # one-hot 编码 (3797, 3)
    print(imgMulti.shape, samplesTrain.shape, categoryTrain.shape, labelsMat.shape)
    # 图像所有像素点都作为预测样本集的输入值 samplesTest
    samplesImg = imgMulti[:,:,:nBands].reshape(-1,nBands).astype(np.float32)  # (262144, 4)

    # (4) ANN 分类模型的 创建，配置，训练，检验和预测
    # -- 创建 ANN 模型
    ann = cv.ml.ANN_MLP_create()  # 创建 ANN 模型
    # -- 模型配置和参数设置
    nInputs, nHiddennodes, nOutput = samplesTrain.shape[1], 4, labelsMat.shape[1]  # 4*4*3
    ann.setLayerSizes((nInputs, nHiddennodes, nOutput))  # 设置模型结构：输入层，隐含层，输出层
    ann.setActivationFunction(cv.ml.ANN_MLP_SIGMOID_SYM, 0.75, 1.0)  # 设置激活函数
    ann.setTrainMethod(cv.ml.ANN_MLP_BACKPROP, 0.1, 0.1)  # 设置训练方法和参数
    ann.setTermCriteria(
        (cv.TERM_CRITERIA_MAX_ITER | cv.TERM_CRITERIA_EPS, 1000, 0.01))  # 设置收敛准则
    # -- 模型训练
    ann.train(samplesTrain, cv.ml.ROW_SAMPLE, labelsMat)  # 训练 ANN 模型，(3797,nClasses=3)
    # -- 模型测试
    nTest = samplesTrain.shape[0]
    _, result = ann.predict(samplesTrain)  # 模型预测样本的分类结果，(3797,nClasses=3)
    predTest = result.argmax(axis=1).reshape((-1, 1))  # 将分类结果转换为类别序号 (3797,1)
    matches = (predTest == categoryTrain)  # 比较模型预测结果与样本标签
    correct = np.count_nonzero(matches)  # 预测正确的样本数量
    accuracy = 100 * correct / nTest  # 模型预测的准确率
    print("ANN model with multi outputs:")
    print("\tnInputs={}, nHiddennodes={}, nOutput={}".format(nInputs, nHiddennodes, nOutput))
    print("\tsamples={}, correct={}, accuracy={:.2f}%".format(nTest, correct, accuracy))
    # -- 模型预测，用训练好的模型进行分类，并与正确结果进行比较
    _, result = ann.predict(samplesImg)  # 模型预测样本的分类结果，(262144,4)->(262144,3)
    predImg = result.argmax(axis=1).reshape((-1, 1))  # 将分类结果转换为类别序号 (262144,1)

    # (5) 统计分类结果
    unique, count = np.unique(predImg, return_counts=True)  # 统计各类别的数量
    imgClassify = predImg.reshape((height, width)).astype(np.uint8)  # 恢复为二维 (512, 512)
    print("Classification of spectrum by ANN:")
    for k in range(3):
        print("\tType {}: {}".format(unique[k], count[k]))

    # (6) 显示标记的分类结果
    plt.figure(figsize=(9, 3.5))
    plt.subplot(131), plt.axis('off'), plt.title("(1) Class_0: Urban region")
    imgC0 = np.ones((height, width, 3), np.uint8)*64  # 第 1 类
    imgC0[imgClassify==0] = (250,206,135)  # 模型预测区域
    imgC0[imgLabel==1] = (255,0,0)  # 确定标记区域
    plt.imshow(cv.cvtColor(imgC0, cv.COLOR_BGR2RGB))
    plt.subplot(132), plt.axis('off'), plt.title("(2) Class_1: Vegetation region")
    imgC1 = np.ones((height, width, 3), np.uint8)*64  # 第 2 类
    imgC1[imgClassify==1] = (143,188,143)  # 模型预测区域
    imgC1[imgLabel==2] = (0,255,0)  # 确定标记区域
    plt.imshow(cv.cvtColor(imgC1, cv.COLOR_BGR2RGB))
    plt.subplot(133), plt.axis('off'), plt.title("(3) Class_2: Water region")
    imgC2 = np.ones((height, width, 3), np.uint8)*64  # 第 3 类
    imgC2[imgClassify==2] = (203,192,255)  # 模型预测区域
    imgC2[imgLabel==3] = (0,0,255)  # 确定标记区域
    plt.imshow(cv.cvtColor(imgC2, cv.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()
