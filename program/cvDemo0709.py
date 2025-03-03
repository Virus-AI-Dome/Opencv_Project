"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【0709】图像的色阶自动调整算法
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# 手动调整色阶
def levelsAdjust(img, Sin=0, Hin=255, Mt=1.0, Sout=0, Hout=255):
    Sin = min(max(Sin, 0), Hin - 2)  # Sin, 黑场阈值, 0<=Sin<Hin
    Hin = min(Hin, 255)  # Hin, 白场阈值, Sin<Hin<=255
    Mt = min(max(Mt, 0.01), 9.99)  # Mt, 灰场调节值, 0.01~9.99
    Sout = min(max(Sout, 0), Hout - 2)  # Sout, 输出黑场阈值, 0<=Sout<Hout
    Hout = min(Hout, 255)  # Hout, 输出白场阈值, Sout<Hout<=255
    difIn = Hin - Sin
    difOut = Hout - Sout
    table = np.zeros(256, np.uint16)
    for i in range(256):
        V1 = min(max(255 * (i - Sin) / difIn, 0), 255)  # 输入动态线性拉伸
        V2 = 255 * np.power(V1 / 255, 1 / Mt)  # 灰场伽马调节
        table[i] = min(max(Sout + difOut * V2 / 255, 0), 255)  # 输出线性拉伸
    imgTone = cv.LUT(img, table)
    return imgTone

# 自动调整色阶
def autoLevels(gray, cutoff=0.1):
    table = np.zeros((1, 256), np.uint8)
    # cutoff=0.1, 计算 0.1%, 99.9% 分位的灰度值
    low = np.percentile(gray, q=cutoff)  # cutoff=0.1, 0.1 分位的灰度值
    high = np.percentile(gray, q=100 - cutoff)  # 99.9 分位的灰度值, [0, high] 占比99.9%
    # 输入动态线性拉伸
    Sin = min(max(low, 0), high - 2)  # Sin, 黑场阈值, 0<=Sin<Hin
    Hin = min(high, 255)  # Hin, 白场阈值, Sin<Hin<=255
    difIn = Hin - Sin
    V1 = np.array([(min(max(255 * (i - Sin) / difIn, 0), 255)) for i in range(256)])
    # 灰场伽马调节
    gradMed = np.median(gray)  # 拉伸前的中值
    Mt = V1[int(gradMed)] / 128.  # 拉伸后的映射值
    V2 = 255 * np.power(V1 / 255, 1 / Mt)  # 伽马调节
    # 输出线性拉伸
    Sout, Hout = 5, 250  # Sout 输出黑场阈值, Hout 输出白场阈值
    difOut = Hout - Sout
    table[0, :] = np.array([(min(max(Sout + difOut * V2[i] / 255, 0), 255)) for i in range(256)])
    return cv.LUT(gray, table)

if __name__ == '__main__':
    # Photoshop 自动色阶调整算法
    gray = cv.imread("../images/Fig0704.png", flags=0)  # 读取为灰度图像
    print("cutoff={}, minG={}, maxG={}".format(0.0, gray.min(), gray.min()))

    # 色阶手动调整
    equManual = levelsAdjust(gray, 64, 200, 0.8, 10, 250)  # 手动调节
    # 色阶自动调整
    cutoff = 0.1  # 截断比例, 范围 [0.0,1.0]
    equAuto = autoLevels(gray, cutoff)

    plt.figure(figsize=(9, 3.5))
    plt.subplot(131), plt.title("(1) Original"), plt.axis('off')
    plt.imshow(cv.cvtColor(gray, cv.COLOR_BGR2RGB))
    plt.subplot(132), plt.title("(2) ManualTuned"), plt.axis('off')
    plt.imshow(cv.cvtColor(equManual, cv.COLOR_BGR2RGB))
    plt.subplot(133), plt.title("(3) AutoLevels"), plt.axis('off')
    plt.imshow(cv.cvtColor(equAuto, cv.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()
