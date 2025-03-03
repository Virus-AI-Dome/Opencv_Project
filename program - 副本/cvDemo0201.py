"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【0201】图像属性与数据格式
import cv2 as cv
import numpy as np

if __name__ == '__main__':
    # 读取图像，支持 bmp、jpg、png、tiff 等常用格式
    filepath = "../images/imgLena.tif"  # 读取文件的路径
    img = cv.imread(filepath, flags=1)  # flags=1 读取彩色图像(BGR)
    gray = cv.imread(filepath, flags=0)  # flags=0 读取为灰度图像

    # 维数(ndim)，形状(shape)，元素总数(size)，数据类型(dtype)
    print("Ndim of img(BGR): {}, gray: {}".format(img.ndim, gray.ndim))
    print("Shape of img(BGR): {}, gray: {}".format(img.shape, gray.shape))  # number of rows, columns and channels
    print("Size of img(BGR): {}, gray: {}".format(img.size, gray.size))  # size = rows * columns * channels

    imgFloat = img.astype(np.float32) / 255
    print("Dtype of img(BGR): {}, gray: {}".format(img.dtype, gray.dtype))  # uint8
    print("Dtype of imgFloat: {}".format(imgFloat.dtype))  # float32


