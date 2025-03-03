"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""


# 【1602】特征描述之傅里叶描述符
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def fftDescribe(s):  # 计算边界 s 的傅里叶描述符
    sComplex = np.empty((s.shape[0], 1, 2), np.int32)  # 声明二维数组 (1816,1,2)
    sComplex[:, 0, :] = s[:, :]  # xk (:,:,0), yk (:,:,1)
    # 中心化, centralized 2d array f(x,y) * (-1)^(x+y)
    mask = np.ones(sComplex.shape)  # (1816, 1, 2)
    mask[1::2, ::2] = -1  # 中心化
    mask[::2, 1::2] = -1
    sCent = sComplex * mask  # f(x,y) * (-1)^(x+y)  # (1816, 1, 2)
    sDft = np.empty(sComplex.shape)  # (1816, 1, 2)
    cv.dft(sCent, sDft, cv.DFT_COMPLEX_INPUT + cv.DFT_COMPLEX_OUTPUT)  # 傅里叶变换
    return sDft

def reconstruct(sDft, scale, size, ratio=1.0):  # 由傅里叶描述符重建轮廓图
    K = sDft.shape[0]  # 傅里叶描述符的总长度
    pLowF = int(K * ratio)  # 保留的低频系数的长度
    low, high = int(K/2) - int(pLowF/2), int(K/2) + int(pLowF/2)
    sDftLow = np.zeros(sDft.shape, np.float64)  # [0,low) 和 (high, K] 区间置 0
    sDftLow[low:high, :, :] = sDft[low:high, :, :]  # 保留 [low,high] 区间的傅里叶描述符

    iDft = np.empty(sDftLow.shape)  # (1816, 1, 2)
    cv.idft(sDftLow, iDft, cv.DFT_COMPLEX_INPUT | cv.DFT_COMPLEX_OUTPUT)  # 傅里叶逆变换
    # 去中心化, centralized 2d array g(x,y) * (-1)^(x+y)
    mask2 = np.ones(iDft.shape, np.int32)  # (1816, 1, 2)
    mask2[1::2, ::2] = -1  # 去中心化
    mask2[::2, 1::2] = -1
    idftCent = iDft * mask2  # g(x,y) * (-1)^(x+y)
    if idftCent.min() < 0:
        idftCent -= idftCent.min()
    idftCent *= scale / idftCent.max()  # 调整尺度比例
    sRebuild = np.squeeze(idftCent).astype(np.int32)  # (1816, 1, 2)->(1816,2)
    print("ratio:{}\tdescriptor:{}\t max/min:{}/{}".format(ratio, pLowF, sRebuild.max(), sRebuild.min()))

    rebuild = np.ones(size, np.uint8) * 255  # 创建空白图像
    cv.rectangle(rebuild, (2, 2), (size[0]-2, size[1]-2), (0,0,0), 2)  # 绘制边框
    cv.polylines(rebuild, [sRebuild], True, 0, thickness=2)  # 绘制多边形，闭合曲线
    return rebuild

if __name__ == '__main__':
    img = cv.imread("../images/Fig1602.png", flags=1)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 灰度图像
    print("shape of image:", gray.shape)  # (600, 600)

    _, binary = cv.threshold(gray, 200, 255, cv.THRESH_BINARY|cv.THRESH_OTSU)
    # 寻找二值化图中的轮廓，method=cv.CHAIN_APPROX_NONE 输出轮廓的每个像素点
    contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)  # OpenCV4~
    cnts = sorted(contours, key=cv.contourArea, reverse=True)  # 所有轮廓按面积排序
    cnt = cnts[0]  # 第 0 个轮廓，面积最大的轮廓，(1816, 1, 2)
    cntPoints = np.squeeze(cnt)  # 删除维度为 1 的数组维度，(1816, 1, 2)->(1816,2)
    imgCnts = np.zeros(gray.shape[:2], np.uint8)  # 创建空白图像
    cv.drawContours(imgCnts, cnt, -1, (255, 255, 255), 2)  # 绘制轮廓

    # 计算傅里叶描述符
    if (cntPoints.shape[0] % 2):  # 如果轮廓像素为奇数则补充为偶数
        cntPoints = np.append(cntPoints, [cntPoints[0]], axis=0)  # 首尾循环，补为偶数 (1816, 2)
    K = cntPoints.shape[0]  # 轮廓点的数量
    scale = cntPoints.max()  # 尺度系数
    sDft = fftDescribe(cntPoints)  # 复数数组，保留全部系数，(1816, 1, 2)
    print("cntPoint:", cntPoints.shape, "scale:", cntPoints.max(), cntPoints.min())

    # 由全部傅里叶描述符 重建轮廓曲线
    size = gray.shape[:2]
    rebuild = reconstruct(sDft, scale, size)  # 由全部傅里叶描述子重建轮廓曲线 (1816,)
    # 由低频傅里叶描述符 重建轮廓曲线，删除高频系数
    kReb = [0.1, 0.05, 0.02, 0.01, 0.005]
    rebuild1 = reconstruct(sDft, scale, size, ratio=kReb[0])  # 低频系数 (181,2)
    rebuild2 = reconstruct(sDft, scale, size, ratio=kReb[1])  # 低频系数 (90,2)
    rebuild3 = reconstruct(sDft, scale, size, ratio=kReb[2])  # 低频系数 (36,2)
    rebuild4 = reconstruct(sDft, scale, size, ratio=kReb[3])  # 低频系数 (18,2)
    rebuild5 = reconstruct(sDft, scale, size, ratio=kReb[4])  # 低频系数 (9,2)

    plt.figure(figsize=(9, 5.4))
    plt.subplot(241), plt.axis('off'), plt.title("(1) Original")
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.subplot(242), plt.axis('off'), plt.title("(2) Contour")
    plt.imshow(cv.cvtColor(imgCnts, cv.COLOR_BGR2RGB))
    plt.imshow(imgCnts, cmap='gray')
    plt.subplot(243), plt.axis('off'), plt.title("(3) Recovery (100%)")
    plt.imshow(rebuild, cmap='gray')
    for i in range(len(kReb)):
        plt.subplot(2,4,4+i), plt.axis('off')
        plt.title("({}) Rebuild{} ({}%)".format(i+4, i+1, kReb[i]*100))
        plt.imshow(eval("rebuild{}".format(i+1)), cmap='gray')
    # plt.subplot(244), plt.axis('off'), plt.title("Rebuild1 (10%)")
    # plt.imshow(rebuild1, cmap='gray')
    # plt.subplot(245), plt.axis('off'), plt.title("Rebuild2 (5.0%)")
    # plt.imshow(rebuild2, cmap='gray')
    # plt.subplot(246), plt.axis('off'), plt.title("Rebuild3 (2.5%)")
    # plt.imshow(rebuild3, cmap='gray')
    # plt.subplot(247), plt.axis('off'), plt.title("Rebuild4 (1.0%)")
    # plt.imshow(rebuild4, cmap='gray')
    # plt.subplot(248), plt.axis('off'), plt.title("Rebuild5 (0.5%)")
    # plt.imshow(rebuild5, cmap='gray')
    plt.tight_layout()
    plt.show()