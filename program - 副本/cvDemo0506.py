"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【0506】提取和嵌入目标图像
import cv2 as cv
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv.imread("../images/Lena.tif")  # 读取彩色图像(BGR)
    logo = cv.imread("../images/logoCV.png")  # 读取目标图像
    h2, w2= logo.shape[:2]  # Logo 图像尺寸
    imgROI = img[0:h2, 0:w2]  # 从图像中裁剪叠放区域

    # (1) 灰度化、二值化，生成 Logo 掩模图像
    gray = cv.cvtColor(logo, cv.COLOR_BGR2GRAY)  # Logo 转为灰度图像
    _, mask = cv.threshold(gray, 175, 255, cv.THRESH_BINARY_INV)  # 二值处理得到掩模图像
    # (2) 带掩模的位操作，生成合成图像的背景和前景


    background = cv.bitwise_and(imgROI, imgROI, mask=cv.bitwise_not(mask))  # 生成合成背景
    frontground = cv.bitwise_and(logo, logo, mask=mask)  # 生成合成前景

    # (3) 由前景和背景合成叠加图像
    compositeROI = cv.add(background, frontground)   # 前景与背景相加，得到叠加 ROI

    composite = img.copy()
    composite[0:h2,0:w2] = compositeROI  # 叠加 Logo 的合成图像

    # (4) 对照方法：替换方法添加 Logo
    replace = img.copy()
    replace[0:h2,0:w2] = logo[:,:]
    # (5) 对照方法：通过加法添加 Logo
    cvAdd = img.copy()
    cvAdd[0:h2,0:w2] = cv.add(imgROI, logo)

    plt.figure(figsize=(9, 6))
    plt.subplot(231), plt.title("(1) replace"), plt.axis('off')
    plt.imshow(cv.cvtColor(replace, cv.COLOR_BGR2RGB))
    plt.subplot(232), plt.title("(2) cv.add"), plt.axis('off')
    plt.imshow(cv.cvtColor(cvAdd, cv.COLOR_BGR2RGB))
    plt.subplot(233), plt.title("(3) composite"), plt.axis('off')
    plt.imshow(cv.cvtColor(composite, cv.COLOR_BGR2RGB))
    plt.subplot(234), plt.title("(4) mask"), plt.axis('off')
    plt.imshow(mask, 'gray')
    plt.subplot(235), plt.title("(5) background"), plt.axis('off')
    plt.imshow(cv.cvtColor(background, cv.COLOR_BGR2RGB))
    plt.subplot(236), plt.title("(6) frontground"), plt.axis('off')
    plt.imshow(cv.cvtColor(frontground, cv.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()
