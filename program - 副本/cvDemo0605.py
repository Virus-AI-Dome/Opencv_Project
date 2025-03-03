"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【0605】图像的斜切 (扭变)
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv.imread("../images/Fig0601.png")  # 读取彩色图像(BGR)
    height, width = img.shape[:2]  # 图片的高度和宽度

    angle = 20 * np.pi/180  # 斜切角度
    # (1) 水平斜切
    MAS = np.float32([[1, np.tan(angle), 0], [0, 1, 0]])  # 斜切变换矩阵
    wShear = width + int(height*abs(np.tan(angle)))  # 调整宽度
    imgShearH = cv.warpAffine(img, MAS, (wShear, height))
    # (2) 垂直斜切
    MAS = np.float32([[1, 0, 0], [np.tan(angle), 1, 0]])  # 斜切变换矩阵
    hShear = height + int(width*abs(np.tan(angle)))  # 调整高度
    imgShearV = cv.warpAffine(img, MAS, (width, hShear))

    print(img.shape, imgShearH.shape, imgShearV.shape)
    plt.figure(figsize=(9, 4))
    plt.subplot(131), plt.axis('off'), plt.title("(1) Original")
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.subplot(132), plt.axis('off'), plt.title("(2) Horizontal shear")
    plt.imshow(cv.cvtColor(imgShearH, cv.COLOR_BGR2RGB))
    plt.subplot(133), plt.axis('off'), plt.title("(3) Vertical shear")
    plt.imshow(cv.cvtColor(imgShearV, cv.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()






