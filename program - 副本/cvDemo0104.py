"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【0104】OpenCV 图像窗口中显示图像
import cv2 as cv

if __name__ == '__main__':
    filepath = "../images/Lena.tif"  # 读取文件的路径
    img = cv.imread(filepath, flags=1)  # flags=1 读取彩色图像(BGR)
    gray = cv.imread(filepath, flags=0)  # flags=0 读取为灰度图像

    cv.imshow("Lena", img)  # 在窗口 img1 显示图像
    cv.imshow("Lena_gray", gray)  # 在窗口 img2 显示图像
    key = cv.waitKey(0)  # delay=0, 不自动关闭
    cv.destroyAllWindows()
