"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【1709】特征匹配之最近邻匹配（FLANN）
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

    # (3) 特征点匹配，KLANN-KnnMatch 返回两个匹配点，最优点和次优点
    indexParams = dict(algorithm=1, trees=5)  # 设置 KD-TREE 算法和参数
    searchParams = dict(checks=100)  # 设置递归搜索层数
    flann = cv.FlannBasedMatcher(indexParams, searchParams)  # 创建 FLANN 匹配器
    # matches = flann.match(desRef, desObj)
    matches = flann.knnMatch(desRef, desObj, k=2)  # FLANN 匹配，返回最优点和次优点 2个结果
    good = []  # 筛选匹配结果
    for i, (m, n) in enumerate(matches):
        if m.distance<0.8*n.distance:  # 最优点距离/次优点距离 之比小于阈值0.8
            good.append(m)  # 保留显著性高度匹配结果
    matches1 = cv.drawMatches(imgRef, kpRef, imgObj, kpObj, good, None, matchColor=(0,255,0), flags=0)
    print("(1) FLANNmatches:{}, goodMatches:{}".format(len(matches1), len(good)))

    # (3) 单应性映射筛选匹配结果
    # 从 imgRef 框选目标区域，也可以直接设置
    # (x, y, w, h) = cv.selectROI(imgRef, showCrosshair=True, fromCenter=False)
    (x, y, w, h) = 316, 259, 116, 43
    print("ROI: x={}, y={}, w={}, h={}".format(x, y, w, h))
    rectPoints = [[x,y], [x+w,y], [x+w,y+h], [x,y+h]]  # 框选区域的顶点坐标 (x,y)
    pts = np.float32(rectPoints).reshape(-1,1,2)  # imgRef 中的指定区域
    cv.polylines(imgRef, [np.int32(pts)], True, (255,0,0), 3)  # 在 imbRef 绘制框选区域
    if len(good) > 10:  # MIN_MATCH_COUNT=10
        refPts = np.float32([kpRef[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)  # 关键点坐标
        objPts = np.float32([kpObj[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        Mat, mask = cv.findHomography(refPts, objPts, cv.RANSAC, 5.0)  # 单映射变换矩阵
        matchesMask = mask.ravel().tolist()  # 展平并转为列表
        ptsTrans = cv.perspectiveTransform(pts, Mat)  # 投影变换计算，imgObj 中的指定区域
        cv.polylines(imgObj, [np.int32(ptsTrans)], True, (255,0,0), 3)  # 在 imbObj 绘制映射边框
    else:
        print("Not enough matches.")
    # print("Rect points in imgRef:", pts)
    # print("Rect points in imgObj:", ptsTrans)
    # 绘制匹配结果
    matches2 = cv.drawMatches(imgRef, kpRef, imgObj, kpObj, good, None, matchColor=(0,255,0))
    print("(2) FLANNmatches:{}, goodMatches:{}, filteredMatches:{}".format(len(matches2), len(good), np.sum(mask)))

    # (4) 筛选并绘制指定区域内的匹配结果
    roiGood = []
    for i in range(len(good)):
        (xi, yi) = kpRef[good[i].queryIdx].pt
        if x<xi<x+w and y<yi<y+h and matchesMask[i]==True:
            roiGood.append(good[i])
    matches3 = cv.drawMatches(imgRef, kpRef, imgObj, kpObj, roiGood, None, matchColor=(0,255,0))
    print("(3) FLANNmatches:{}, goodMatches:{}, roiMatches:{}".format(len(matches2), len(good), len(roiGood)))

    plt.figure(figsize=(9, 6))
    plt.subplot(211), plt.axis('off'), plt.title("(1) FLANN with homography")
    plt.imshow(cv.cvtColor(matches2, cv.COLOR_BGR2RGB))  # FLANN with homography
    plt.subplot(212), plt.axis('off'), plt.title("(2) FLANN inside the ROI")
    plt.imshow(cv.cvtColor(matches3, cv.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()
