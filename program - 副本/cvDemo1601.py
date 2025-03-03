"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【1601】特征描述之弗里曼链码
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def FreemanChainCode(cLoop, gridsep=1):  # 由闭合边界点集生成弗里曼链码
    # Freeman 8 方向链码的方向数
    dictFreeman = {(1,0):0, (1,1):1, (0,1):2, (-1,1):3, (-1,0):4, (-1,-1):5, (0,-1):6, (1,-1):7}
    diffCloop = np.diff(cLoop, axis=0) // gridsep  # cLoop 的一阶差分码，(k+1,2)->(k,2)
    direction = [tuple(x) for x in diffCloop.tolist()]
    codeList = list(map(dictFreeman.get, direction))  # 查字典获得链码，k
    code = np.array(codeList)  # 转回 Numpy 数组，(k,)
    return code

def boundarySubsample(points, gridsep):  # 对闭合边界曲线向下降采样
    gridsep = max(int(gridsep), 2)  # gridsep 为整数
    pointsGrid = points.copy()  # 初始化边界点的栅格坐标
    subPointsList = []  # 初始化降采样点集
    Grid = np.zeros((4, 2), np.int16)  # 初始化格栅顶点坐标
    dist2Grid = np.zeros((4,), np.float64)  # 初始化边界点到栅格顶点的距离
    for i in range(points.shape[0]):  # 遍历边界点
        [xi, yi] = points[i,:]  # 第 i 个边界点的坐标
        [xgrid, ygrid] = [xi-xi%gridsep, yi-yi%gridsep]  # 边界点[xi,yi] 所属栅格的顶点坐标
        Grid[0,:] = [xgrid, ygrid]  # 栅格的上下左右 4 个顶点
        Grid[1,:] = [xgrid, ygrid+gridsep]
        Grid[2,:] = [xgrid+gridsep, ygrid]
        Grid[3,:] = [xgrid+gridsep, ygrid+gridsep]
        # dist2Grid[:] = [np.linalg.norm(points[i,:] - Grid[k,:]) for k in range(4)]  # 边界点到格栅各顶点距离
        dist2Grid[:] = [np.sqrt(np.sum(np.square(points[i,:] - Grid[k,:]))) for k in range(4)]
        GridMin = np.argmax(-dist2Grid, axis=0)  # 最小值索引，最近栅格顶点的编号
        pointsGrid[i,:] = Grid[GridMin,:]  # 边界点被吸引到最近的栅格顶点
        if (pointsGrid[i,:] != pointsGrid[i-1,:]).any():  # 相邻边界点栅格坐标是否相同
            subPointsList.append(pointsGrid[i])  # 只添加不同的点，即删除重复的边界点
    subPoints = np.array(subPointsList)
    return subPoints

if __name__ == '__main__':
    img = cv.imread("../images/Fig1601.png", flags=1)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 灰度图像
    blur = cv.boxFilter(gray, -1, (3, 3))  # 盒式滤波器，3*3 平滑核
    _, binary = cv.threshold(blur, 200, 255, cv.THRESH_OTSU + cv.THRESH_BINARY_INV)
    # 寻找二值化图中的轮廓，**method=cv.CHAIN_APPROX_NONE 输出轮廓的每个像素点！**
    contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)  # OpenCV4~
    # 绘制全部轮廓，contourIdx=-1 绘制全部轮廓
    imgCnts = np.zeros(gray.shape[:2], np.uint8)  # 绘制轮廓函数会修改原始图像
    imgCnts = cv.drawContours(imgCnts, contours, -1, (255,255,255), thickness=2)  # 绘制全部轮廓

    # 获取最大轮廓
    cnts = sorted(contours, key=cv.contourArea, reverse=True)  # 所有轮廓按面积排序
    cnt = cnts[0]  # 第 0 个轮廓，面积最大的轮廓，(1458, 1, 2)
    cntPoints = np.squeeze(cnt)  # 删除维度为1的数组维度，(1458,1,2)->(1458,2)
    maxContour = np.zeros(gray.shape[:2], np.uint8)  # 初始化最大轮廓图像
    cv.drawContours(maxContour, cnt, -1, (255, 255, 255), thickness=2)  # 绘制轮廓 cnt
    print("len(contours) =", len(contours))  # contours 所有轮廓的列表
    print("area of max contour: ", cv.contourArea(cnt))  # 轮廓面积
    print("perimeter of max contour: {:.1f}".format(cv.arcLength(cnt, True)))  # 轮廓周长

    # 向下降采样，简化轮廓的边界
    gridsep = 25  # 采样间隔
    subPoints = boundarySubsample(cntPoints, gridsep)  # 自定义函数，通过向下采样简化轮廓
    print("points of contour:", cntPoints.shape[0])  # 原始轮廓点数：1458
    print("subsample steps: {}, points of subsample: {}".format(gridsep,subPoints.shape[0]))  # 降采样轮廓点数 81
    subContour1 = np.zeros(gray.shape[:2], np.uint8)  # 初始化简化轮廓图像
    [cv.circle(subContour1, (point[0],point[1]), 1, 160, -1) for point in cntPoints]  # 绘制初始轮廓的采样点
    [cv.circle(subContour1, (point[0],point[1]), 4, 255, -1) for point in subPoints]  # 绘制降采样轮廓的采样点
    cv.polylines(subContour1, [subPoints], True, 255, thickness=2)  # 绘制多边形，闭合曲线

    # 向下降采样，简化轮廓的边界
    gridsep = 50  # 采样间隔
    subPoints = boundarySubsample(cntPoints, gridsep)  # 自定义函数，通过向下采样简化轮廓
    print("subsample steps: {}, points of subsample:{}".format(gridsep,subPoints.shape[0]))  # 降采样轮廓点数 40
    subContour2 = np.zeros(gray.shape[:2], np.uint8)  # 初始化简化轮廓图像
    [cv.circle(subContour2, (point[0],point[1]), 1, 160, -1) for point in cntPoints]  # 绘制初始轮廓的采样点
    [cv.circle(subContour2, (point[0],point[1]), 4, 255, -1) for point in subPoints]  # 绘制降采样轮廓的采样点
    cv.polylines(subContour2, [subPoints], True, 255, thickness=2)  # 绘制多边形，闭合曲线

    # 生成 Freeman 链码
    cntPoints = np.squeeze(cnt)  # 删除维度为1 的数组维度，(1458,1,2)->(1458,2)
    # pointsLoop = np.append(cntPoints, [cntPoints[0]], axis=0)  # 首尾循环，结尾添加 cntPoints[0]
    pointsLoop = np.append(cntPoints, [cntPoints[0]], axis=0)  # 首尾循环，结尾添加 cntPoints[0]
    chainCode = FreemanChainCode(pointsLoop, gridsep=1)  # 自定义函数，生成链码 (1458,)
    print("Freeman chain code:", chainCode.shape)  # 链码长度为轮廓长度 1458
    # print("subsample steps: {}, points of subsample:{}".format(gridsep, subPoints.shape[0]))
    if (subPoints[0]==subPoints[-1]).all():
        subPointsLoop = subPoints  # 首尾相同，不需要构造循环 (40,2)
    else:
        subPointsLoop = np.append(subPoints, [subPoints[0]], axis=0)  # 首尾循环
    subChainCode = FreemanChainCode(subPointsLoop, gridsep=50)  # 自定义函数，生成链码 (40,)
    print("Down-sampling Freeman chain code:", subChainCode.shape)  # 链码长度为简化轮廓长度
    print(subChainCode)

    plt.figure(figsize=(10, 6))
    plt.subplot(231), plt.title("(1) Original")
    plt.axis('off'), plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.subplot(232), plt.title("(2) Binary image")
    plt.axis('off'), plt.imshow(binary, 'gray')
    plt.subplot(233), plt.title("(3) Contours")
    plt.axis('off'), plt.imshow(imgCnts, 'gray')
    plt.subplot(234), plt.title("(4) Max contour")
    plt.axis('off'), plt.imshow(maxContour, 'gray')
    plt.subplot(235), plt.title("(5) DownSampling(grid=25)")
    plt.axis('off'), plt.imshow(subContour1, 'gray')
    plt.subplot(236), plt.title("(6) DownSampling(grid=50)")
    plt.axis('off'), plt.imshow(subContour2, 'gray')
    plt.tight_layout()
    plt.show()
