"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【0602】图像的缩放
import cv2 as cv
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv.imread("../images/Lena.tif")  # 读取彩色图像(BGR)
    imgZoom1 = cv.resize(img, (600, 480))
    imgZoom2 = cv.resize(img, None, fx=1.2, fy=0.8, interpolation=cv.INTER_CUBIC)

    plt.figure(figsize=(9, 3.3))
    plt.subplot(131), plt.title("(1) Original"), plt.axis('off')
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.subplot(132), plt.title("(2) Zoom 1")
    plt.imshow(cv.cvtColor(imgZoom1, cv.COLOR_BGR2RGB))
    plt.subplot(133), plt.title("(3) Zoom 2")
    plt.imshow(cv.cvtColor(imgZoom2, cv.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()




