"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【0301】图像的颜色空间转换
import cv2 as cv
from matplotlib import pyplot as plt

if __name__ == '__main__':
    imgBGR = cv.imread("../images/Lena.tif", flags=1)  # 读取为彩色图像

    imgRGB = cv.cvtColor(imgBGR, cv.COLOR_BGR2RGB)  # BGR 转换为 RGB
    imgGRAY = cv.cvtColor(imgBGR, cv.COLOR_BGR2GRAY)  # BGR 转灰度图像
    imgHSV = cv.cvtColor(imgBGR, cv.COLOR_BGR2HSV)  # BGR 转 HSV 图像
    imgYCrCb = cv.cvtColor(imgBGR, cv.COLOR_BGR2YCrCb)  # BGR 转 YCrCb
    imgHLS = cv.cvtColor(imgBGR, cv.COLOR_BGR2HLS)  # BGR 转 HLS 图像
    imgXYZ = cv.cvtColor(imgBGR, cv.COLOR_BGR2XYZ)  # BGR 转 XYZ 图像
    imgLAB = cv.cvtColor(imgBGR, cv.COLOR_BGR2LAB)  # BGR 转 LAB 图像
    imgYUV = cv.cvtColor(imgBGR, cv.COLOR_BGR2YUV)  # BGR 转 YUV 图像

    # 调用matplotlib显示处理结果
    titles = ['BGR', 'RGB', 'GRAY', 'HSV', 'YCrCb', 'HLS', 'XYZ', 'LAB', 'YUV']
    images = [imgBGR, imgRGB, imgGRAY, imgHSV, imgYCrCb,
              imgHLS, imgXYZ, imgLAB, imgYUV]
    plt.figure(figsize=(10, 8))
    for i in range(9):
        plt.subplot(3, 3, i+1), plt.imshow(cv.cvtColor(images[i],cv.COLOR_BGR2RGB))
        plt.title("({}) {}".format(i+1, titles[i]))
        plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.show()
