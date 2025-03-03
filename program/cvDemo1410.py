"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""


# 【1410】图像的矩与不变矩
import cv2 as cv

if __name__ == '__main__':
    gray = cv.imread("../images/Fig1402.png", flags=0)
    _, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY_INV)

    # (1) 图像的矩
    grayMmoments = cv.moments(gray)  # 返回字典 Mu，几何矩 mpq, 中心矩 mupq 和归一化矩 nupq
    grayHuM = cv.HuMoments(grayMmoments)  # 计算 Hu 不变矩
    print(type(grayMmoments), type(grayHuM), grayHuM.shape)
    print("Moments of gray:\n", grayMmoments)
    print("HuMoments of gray:\n", grayHuM)

    # (2) 轮廓的矩（点坐标向量数组）
    contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)  # OpenCV4~
    cnt = contours[0]  # 轮廓，点坐标向量数组 (30, 1, 2)
    cntMoments = cv.moments(cnt)  # 返回字典 Mu
    cntHuM = cv.HuMoments(cntMoments)  # 计算 Hu 不变矩
    print("Shape of contour:", cnt.shape)
    print("Moments of contour:\n", cntMoments)
    print("HuMoments of contour:\n", cntHuM)

