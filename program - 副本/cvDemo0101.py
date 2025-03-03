"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【0101】OpenCV 读取和保存图像
import cv2 as cv

if __name__ == '__main__':
    # 读取图像，支持 bmp、jpg、png、tiff 等常用格式
    filepath = "../images/Lena.tif"  # 读取文件的路径
    img = cv.imread(filepath, flags=1)  # flags=1 读取彩色图像(BGR)
    gray = cv.imread(filepath, flags=0)  # flags=0 读取为灰度图像

    saveFile = "../images/imgSave1.png"  # 保存文件的路径
    cv.imwrite(saveFile, img, [int(cv.IMWRITE_PNG_COMPRESSION), 8])
    cv.imwrite("../images/imgSave2.png", gray)

