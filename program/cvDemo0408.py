"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【0408】鼠标交互框选矩形区域 (ROI)
import cv2 as cv

if __name__ == '__main__':
    img = cv.imread("../images/Lena.tif",flags=1)  # 读取彩色图像(BGR)
    # 鼠标框选矩形 ROI
    r,g,b = cv.split(img)
    # imgAll = cv.hconcat([r,g,b])
    # cv.imshow('imgAll', imgAll)
    # cv.waitKey(0)
    cv.imshow('Blue Channel', b)  # 显示蓝色通道
    cv.imshow('Green Channel', g)  # 显示绿色通道
    cv.imshow('Red Channel', r)  # 显示红色通道

    # 等待按键关闭窗口
    cv.waitKey(0)
    cv.destroyAllWindows()