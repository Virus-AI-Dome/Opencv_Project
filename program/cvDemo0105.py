"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【0105】使用 Matplotlib 库显示图像
import cv2 as cv
from matplotlib import pyplot as plt

if __name__ == '__main__':
    filepath = "../images/Lena.tif"  # 读取文件的路径
    img = cv.imread(filepath, flags=1)  # flags=1 读取彩色图像(BGR)
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # 图片格式转换：BGR-> RGB
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 图片格式转换：BGR -> Gray

    plt.figure(figsize=(8, 7))  # 创建自定义图像
    plt.subplot(221), plt.title("(1) RGB (Matplotlib)"), plt.axis('off')
    plt.imshow(imgRGB)  # matplotlib 显示彩色图像(RGB格式)
    plt.subplot(222), plt.title("(2) BGR (OpenCV)"), plt.axis('off')
    plt.imshow(img)    # matplotlib 显示彩色图像(BGR格式)
    plt.subplot(223), plt.title("(3) cmap='gray'"), plt.axis('off')
    plt.imshow(gray, cmap='gray')  # matplotlib 显示灰度图像，设置 Gray 参数
    plt.subplot(224), plt.title("(4) without cmap"), plt.axis('off')
    plt.imshow(gray)  # matplotlib 显示灰度图像，未设置 Gray 参数
    plt.tight_layout()  # 自动调整子图间隔
    plt.show()  # 显示图像
