"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【0203】图像的裁剪与拼接
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    # (1) 图像的裁剪
    filepath = "../images/Lena.tif"  # 读取文件的路径
    img = cv.imread(filepath, flags=1)  # flags=1 读取彩色图像(BGR)
    xmin, ymin, w, h = 180, 190, 200, 200  # 裁剪区域的位置：(ymin:ymin+h, xmin:xmin+w)
    imgCrop = img[ymin:ymin+h, xmin:xmin+w].copy()  # 切片获得裁剪后保留的图像区域
    print("img:{}, imgCrop:{}".format(img.shape, imgCrop.shape))

    # (2) 图像的拼接
    logo = cv.imread("../images/Fig0201.png")  # 读取彩色图像(BGR)
    imgH1 = cv.resize(img, (400, 400))  # w=400, h=400
    imgH2 = cv.resize(logo, (300, 400))  # w=300, h=400
    imgH3 = imgH2.copy()
    # stackH = np.hstack((imgH1, imgH2, imgH3))  # Numpy 方法 横向水平拼接
    stackH = cv.hconcat((imgH1, imgH2, imgH3))  # OpenCV方法 横向水平拼接
    print("imgH1:{}, imgH2:{}, imgH3:{}, stackH:{}".format(imgH1.shape, imgH2.shape, imgH3.shape, stackH.shape))
    plt.figure(figsize=(9, 4))
    plt.imshow(cv.cvtColor(stackH, cv.COLOR_BGR2RGB))
    plt.xlim(0, 1000), plt.ylim(400, 0)
    plt.show()

    imgV1 = cv.resize(img, (400, 400))  # w=400, h=400
    imgV2 = cv.resize(logo, (400, 300))  # w=400, h=300
    imgV = (imgV1, imgV2)  # 生成拼接图像的列表或元组
    stackV = cv.vconcat(imgV)  # 多张图像数组的拼接
    # stackV = cv.vconcat((imgV1, imgV2))  # OpenCV方法 纵向垂直拼接
    print("imgV1:{}, imgV2:{}, stackV:{}".format(imgV1.shape, imgV2.shape, stackV.shape))

    cv.imshow("DemoStackH", stackH)  # 在窗口显示图像 stackH
    cv.imshow("DemoStackV", stackV)  # 在窗口显示图像 stackV
    key = cv.waitKey(0)  # 等待按键命令

