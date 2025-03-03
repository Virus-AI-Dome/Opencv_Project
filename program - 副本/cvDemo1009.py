"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【1009】空间滤波之钝化掩蔽
import cv2 as cv
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv.imread("../images/Lena.tif", flags=0)
    print(img.shape[:2])

    # (1) 对原始图像进行高斯平滑
    imgGauss = cv.GaussianBlur(img, (11, 11), sigmaX=5.0)

    # (2) 掩蔽模板：从原始图像中减去平滑图像
    maskPassivate = cv.subtract(img, imgGauss)

    # (3) 掩蔽模板与原始图像相加
    # k<1 减弱钝化掩蔽
    maskWeak = cv.multiply(maskPassivate, 0.5)
    passivation1 = cv.add(img, maskWeak)
    # k=1 钝化掩蔽
    passivation2 = cv.add(img, maskPassivate)
    # k>1 高提升滤波
    maskEnhance = cv.multiply(maskPassivate, 2.0)
    passivation3 = cv.add(img, maskEnhance)

    plt.figure(figsize=(9, 6.5))
    titleList = ["(1) Original", "(2) GaussSBlur", "(3) PassivateMask",
                 "(4) Passivation(k=0.5)", "(5) Passivation(k=1.0)", "(6) Passivation(k=2.0)"]
    imgList = [img, imgGauss, maskPassivate, passivation1, passivation2, passivation3]
    for i in range(6):
        plt.subplot(2,3,i+1), plt.title(titleList[i])
        plt.axis('off'), plt.imshow(imgList[i], 'gray')
    plt.tight_layout()
    plt.show()
