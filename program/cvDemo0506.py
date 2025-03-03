"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""
import cv2
# 【0506】提取和嵌入目标图像
import cv2 as cv
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv.imread("../images/Lena.tif")
    logo = cv.imread("../images/logoCV.png")
    h2, w2, _ = logo.shape
    imgROI = img[0:h2, 0:w2]

    gray = cv.cvtColor(logo, cv.COLOR_BGR2GRAY)
    _, mask = cv.threshold(gray, 175, 255, cv.THRESH_BINARY_INV)
    background = cv.bitwise_and(imgROI, imgROI, mask=cv.bitwise_not(mask))
    frontground = cv.bitwise_and(logo, logo, mask=mask)

    compositeROI = cv.add(background, frontground)

    composite = img.copy()
    composite[0:h2, 0:w2] = compositeROI

    replace = img.copy()
    replace[0:h2, 0:w2] = logo[:,:]

    cvAdd = img.copy()
    cvAdd[0:h2, 0:w2] = cv.add(imgROI, logo)





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

