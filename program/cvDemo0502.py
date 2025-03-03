"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""


# 【0502】带掩模图像的加法运算
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img1 = cv.imread("../images/Lena.tif")  # 读取彩色图像(BGR)
    img2 = cv.imread("../images/Fig0301.png")  # 读取彩色图像(BGR)
    h, w = img1.shape[:2]
    img3 = cv.resize(img2, (w,h))  # 调整图像大小与 img1 相同
    print(img1.shape, img2.shape, img3.shape)
    imgAddCV = cv.add(img1, img3)  # 图像加法 (饱和运算)

    # 掩模加法，矩形掩模图像
    maskRec = np.zeros(img1.shape[:2], np.uint8)  # 生成黑色模板
    xmin, ymin, w, h = 170, 190, 200, 200  # 矩形 ROI 参数，(ymin:ymin+h, xmin:xmin+w)
    maskRec[ymin:ymin+h, xmin:xmin+w] = 255  # 生成矩形掩模图像，ROI 为白色
    imgAddRec = cv.add(img1, img3, mask=maskRec)  # 掩模加法

    # 掩模加法，圆形掩模图像
    maskCir = np.zeros(img1.shape[:2], np.uint8)  # 生成黑色模板
    cv.circle(maskCir, (280,280), 120, 255, -1)  # 生成圆形掩模图像
    imgAddCir = cv.add(img1, img3, mask=maskCir)  # 掩模加法

    plt.figure(figsize=(9, 6))
    plt.subplot(231), plt.title("(1) Original"), plt.axis('off')
    plt.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
    plt.subplot(232), plt.title("(2) Rectangle mask"), plt.axis('off')
    plt.imshow(cv.cvtColor(maskRec, cv.COLOR_BGR2RGB))
    plt.subplot(233), plt.title("(3) Mask addition"), plt.axis('off')
    plt.imshow(cv.cvtColor(imgAddRec, cv.COLOR_BGR2RGB))
    plt.subplot(234), plt.title("(4) Saturation addition"), plt.axis('off')
    plt.imshow(cv.cvtColor(imgAddCV, cv.COLOR_BGR2RGB))
    plt.subplot(235), plt.title("(5) Circular mask"), plt.axis('off')
    plt.imshow(cv.cvtColor(maskCir, cv.COLOR_BGR2RGB))
    plt.subplot(236), plt.title("(6) Mask addition"), plt.axis('off')
    plt.imshow(cv.cvtColor(imgAddCir, cv.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()
