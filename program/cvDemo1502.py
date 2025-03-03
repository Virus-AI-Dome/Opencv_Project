"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

#【1502】SLIC 超像素区域分割
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    # 注意：本例程需要 opencv-contrib-python 包的支持
    img = cv.imread("../images/Lena.tif", flags=1)  # 彩色图像(BGR)
    imgHSV = cv.cvtColor(img, cv.COLOR_BGR2HSV_FULL)  # BGR-HSV 转换

    # (1) SLICO，使用自适应紧致因子进行优化
    region_size = 20
    ruler = 10.0
    slico = cv.ximgproc.createSuperpixelSLIC(imgHSV, cv.ximgproc.SLICO, region_size, ruler)  # 初始化 SLICO
    slico.iterate(5)  # 迭代次数
    slico.enforceLabelConnectivity(50)  # 最小尺寸
    labelSlico = slico.getLabels()  # 超像素标签
    numberSlico = slico.getNumberOfSuperpixels()  # 超像素数目
    maskSlico = slico.getLabelContourMask()  # 获取 Mask，超像素边缘 Mask==1
    maskColor = np.array([maskSlico for i in range(3)]).transpose(1, 2, 0)  # 转为 3 通道
    imgSlico = cv.bitwise_and(img, img, mask=cv.bitwise_not(maskSlico))  # 绘制超像素边界
    imgSlicoW = cv.add(imgSlico, maskColor)
    print("number of SLICO", numberSlico)

    # (2) SLIC，使用所需的区域大小分割图像
    slic = cv.ximgproc.createSuperpixelSLIC(img, cv.ximgproc.SLIC, region_size, ruler)  # 初始化 SLIC
    slic.iterate(5)  # 迭代次数
    slic.enforceLabelConnectivity(50)  # 最小尺寸
    maskSlic = slic.getLabelContourMask()  # 获取 Mask
    imgSlic = cv.bitwise_and(img, img, mask=cv.bitwise_not(maskSlic))  # 绘制超像素边界
    numberSlic = slic.getNumberOfSuperpixels()  # 超像素数目
    print("number of SLIC", numberSlic)

    # (3) MSLIC，使用流形方法进行优化
    region_size = 40
    mslic = cv.ximgproc.createSuperpixelSLIC(imgHSV, cv.ximgproc.MSLIC, region_size, ruler)
    mslic.iterate(5)  # 迭代次数
    mslic.enforceLabelConnectivity(100)  # 最小尺寸
    maskMslic = mslic.getLabelContourMask()  # 获取Mask
    imgMslic = cv.bitwise_and(img, img, mask=cv.bitwise_not(maskMslic))  # 绘制超像素边界
    numberMslic = mslic.getNumberOfSuperpixels()  # 超像素数目
    print("number of MSLIC", numberMslic)

    plt.figure(figsize=(9, 6))
    plt.subplot(231), plt.axis('off'), plt.title("(1) Original")
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))  # 显示 img(RGB)
    plt.subplot(232), plt.axis('off'), plt.title("(2) SLICO mask")
    plt.imshow(maskSlico, 'gray')
    plt.subplot(233), plt.axis('off'), plt.title("(3) SLICO color")
    plt.imshow(cv.cvtColor(imgSlicoW, cv.COLOR_BGR2RGB))
    plt.subplot(234), plt.axis('off'), plt.title("(4) SLIC (SLIC)")
    plt.imshow(cv.cvtColor(imgSlic, cv.COLOR_BGR2RGB))
    plt.subplot(235), plt.axis('off'), plt.title("(5) SLIC (SLICO)")
    plt.imshow(cv.cvtColor(imgSlico, cv.COLOR_BGR2RGB))
    plt.subplot(236), plt.axis('off'), plt.title("(6) SLIC (MSLIC)")
    plt.imshow(cv.cvtColor(imgMslic, cv.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()
