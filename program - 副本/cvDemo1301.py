"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【1301】极坐标系中的圆形图案文字校正
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv.imread("../images/Fig1301.png")  # 读取彩色图像(BGR)
    h, w = img.shape[:2]  # 图片的高度和宽度

    cx, cy = int(w/2), int(h/2)  # 以图像中心点作为变换中心
    maxR = max(cx, cy)  # 最大变换半径
    imgPolar = cv.linearPolar(img, (cx, cy), maxR, cv.INTER_LINEAR)
    imgPR = cv.rotate(imgPolar, cv.ROTATE_90_COUNTERCLOCKWISE)
    imgRebuild = np.hstack((imgPR[:,w//2:], imgPR[:,:w//2]))
    print(img.shape, imgRebuild.shape)

    plt.figure(figsize=(9, 3.5))
    plt.subplot(131), plt.axis('off'), plt.title("(1) Original")
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.subplot(132), plt.axis('off'), plt.title("(2) Polar Transform")
    plt.imshow(cv.cvtColor(imgPolar, cv.COLOR_BGR2RGB))
    plt.subplot(133), plt.axis('off'), plt.title("(3) Polar Rebuild")
    plt.imshow(cv.cvtColor(imgRebuild, cv.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()

