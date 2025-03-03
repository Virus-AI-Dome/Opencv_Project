"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【1708】特征匹配之暴力匹配（BFMatcher）
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    # (1) 读取参考图像
    imgRef = cv.imread("../images/Fig1703a.png", flags=1)
    refer = cv.cvtColor(imgRef, cv.COLOR_BGR2GRAY)  # 参考图像
    height, width = imgRef.shape[:2]  # 图片的高度和宽度
    # 读取或构造检测图像
    imgObj = cv.imread("../images/Fig1703b.png", flags=1)
    object = cv.cvtColor(imgObj, cv.COLOR_BGR2GRAY)  # 目标图像

    # (2) 构造 SIFT 对象，检测关键点，计算特征描述向量
    sift = cv.SIFT.create()  # sift 实例化对象
    kpRef, desRef = sift.detectAndCompute(refer, None)  # 参考图像关键点检测
    kpObj, desObj = sift.detectAndCompute(object, None)  # 检测图像关键点检测
    print("Keypoints: RefImg {}, ObjImg {}".format(len(kpRef), len(kpObj)))

    # (3) 特征点匹配，暴力匹配+交叉匹配筛选，返回最优匹配结果
    bf1 = cv.BFMatcher(crossCheck=True)  # 构造 BFmatcher 对象，设置交叉匹配
    matches = bf1.match(desRef, desObj)  # 对描述符 desRef, desObj 进行匹配
    # matches = sorted(matches, key=lambda x: x.distance)
    imgMatches1 = cv.drawMatches(imgRef, kpRef, imgObj, kpObj, matches[:200], None, matchColor=(0,255,0))
    print("(1) bf.match with crossCheck: {}".format(len(matches)))
    print(type(matches), type(matches[0]))
    print(matches[0].queryIdx, matches[0].trainIdx, matches[0].distance)  # DMatch 的结构和用法

    # (4) 特征点匹配，KNN匹配+比较阈值筛选
    bf2 = cv.BFMatcher()  # 构造 BFmatcher 对象
    matches = bf2.knnMatch(desRef, desObj, k=2)  # KNN匹配，返回最优点和次优点 2个结果
    goodMatches = []  # 筛选匹配结果
    for m, n in matches:  # matches 是元组
        if m.distance < 0.75 * n.distance:  # 最优点距离/次优点距离 之比小于阈值0.75
            goodMatches.append([m])  # 保留显著性高度匹配结果
    # good = [[m] for m, n in matches if m.distance<0.7*n.distance]  # 单行嵌套循环遍历
    imgMatches2 = cv.drawMatchesKnn(imgRef, kpRef, imgObj, kpObj, goodMatches, None, matchColor=(0,255,0))
    print("(2) bf.knnMatch:{}, goodMatch:{}".format(len(matches), len(goodMatches)))  # 400/236
    print(type(matches), type(matches[0]), type(matches[0][0]))
    print(matches[0][0].distance)

    plt.figure(figsize=(9, 6))
    plt.subplot(211), plt.axis('off'), plt.title("(1) BF MinDistMatch")
    plt.imshow(cv.cvtColor(imgMatches1, cv.COLOR_BGR2RGB))
    plt.subplot(212), plt.axis('off'), plt.title("(2) BF KnnMatch")
    plt.imshow(cv.cvtColor(imgMatches2, cv.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()
