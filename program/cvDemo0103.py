"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【0103】读取和保存中文路径的图像
import cv2 as cv
import numpy as np

if __name__ == '__main__':
    filepath = "../images/测试图01.tif"  # 带有中文的文件路径和文件名
    # img1 = cv.imread(filepath, flags=1)  # 中文路径读取失败，但不会报错
    img2 = cv.imdecode(np.fromfile(filepath, dtype=np.uint8), flags=-1)

    saveFile = "../images/测试图02.tif"  # 带有中文的保存文件路径
    # cv.imwrite(saveFile, img2)  # 中文路径保存失败，但不会报错!
    cv.imencode(".jpg", img2)[1].tofile(saveFile)

