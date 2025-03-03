"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【0202】图像的创建和复制
import cv2 as cv
import numpy as np

if __name__ == '__main__':
    # (1) 通过宽度高度值创建 RGB 彩色图像
    height, width, ch = 400, 300, 3  # 行/高度, 列/宽度, 通道数
    imgEmpty = np.empty((height, width, ch), np.uint8)  # 创建空白数组
    imgBlack = np.zeros((height, width, ch), np.uint8)  # 创建黑色图像 R/G/B=0
    imgWhite = np.ones((height, width, ch), np.uint8) * 255  # 创建白色图像 R/G/B=255
    # (2) 创建与已有图像形状相同的新图像
    img = cv.imread("../images/Lena.tif", flags=1)  # flags=1 读取彩色图像(BGR)
    imgBlackLike = np.zeros_like(img)  # 创建与 img 相同形状的黑色图像
    imgWhiteLike = np.ones_like(img) * 255  # 创建与 img 相同形状的白色图像
    # (3) 创建彩色随机图像（R/G/B 为随机数）
    import os
    randomByteArray = bytearray(os.urandom(height * width * ch))  # 产生随机数组
    flatArray = np.array(randomByteArray)  # 转换为 Numpy 一维数组
    imgRGBRand1 = flatArray.reshape(width, height, ch)  # 形状变换为 (w,h,c)
    imgRGBRand2 = flatArray.reshape(height, width, ch)  # 形状变换为 (h,w,c)
    # (4) 创建灰度图像
    grayWhite = np.ones((height, width), np.uint8) * 255  # 创建白色图像 gray=255
    grayBlack = np.zeros((height, width), np.uint8)  # 创建黑色图像 gray=0
    grayEye = np.eye(width)  # 创建对角线元素为1 的单位矩阵
    randomByteArray = bytearray(os.urandom(height * width))  # 产生随机数组
    flatNumpyArray = np.array(randomByteArray)  # 转换为 Numpy 数组
    imgGrayRand = flatNumpyArray.reshape(height, width)  # 创建灰度随机图像
    # (5) 图像的复制
    img1 = img.copy()  # 深拷贝
    img1[:,:,:] = 0  # 修改 img1
    print("img1 is equal to img?", (img1 is img))  # img 随之修改吗？
    img2 = img  # 浅拷贝
    img2[:,:,:] = 0  # 修改 img2
    print("img2 is equal to img?", (img2 is img))  # img 随之修改吗？

    print("Shape of image: gray {}, RGB1 {}, RGB2 {}"
          .format(imgGrayRand.shape, imgRGBRand1.shape, imgRGBRand2.shape))
    cv.imshow("DemoGray", imgGrayRand)  # 在窗口显示 灰度随机图像
    cv.imshow("DemoRGB", imgRGBRand1)  # 在窗口显示 彩色随机图像
    cv.imshow("DemoBlack", imgBlack)  # 在窗口显示 黑色图像
    key = cv.waitKey(0)  # delay=0, 不自动关闭
    cv.destroyAllWindows()
