"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【0503】图像混合与渐变切换
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':

    img1 = cv.imread("../images/Lena.tif")
    img2 = cv.imread("../images/Fig0301.png")
    h, w = img1.shape[:2]
    img3 = cv.resize(img2, (w, h))
    print(img1.shape, img2.shape, img3.shape)
    # 图像加法(饱和加法)
    imgAddCV = cv.add(img1, img3)

    alpha, beta = 0.25, 0.75
    imgAddW1 = cv.addWeighted(img1, alpha, img3, beta, 0)
    alpha, beta = 0.5, 0.5
    imgAddW2 = cv.addWeighted(img1, alpha, img3, beta, 0)
    alpha, beta = 0.75, 0.25
    imgAddW3 = cv.addWeighted(img1, alpha, img3, beta, 0)
    # 实现两幅画的渐变切换
    wList = np.arange(0.0, 1.0, 0.01)
    for weight in wList:
        imgWeight = cv.addWeighted(img1, weight, img3, (1-weight), 0)
        cv.imshow("imgWeight", imgWeight)
        cv.waitKey(50)
    cv.destroyAllWindows()

    plt.figure(figsize=(9, 3.5))
    plt.subplot(1, 3, 1) , plt.title("(1) a=0.2, b=0.8"), plt.axis("off")
    plt.imshow(cv.cvtColor(imgAddW1, cv.COLOR_BGR2RGB))
    plt.subplot(1,3,2) , plt.title("(2) a=0.5, b=0.5"), plt.axis("off")
    plt.imshow(cv.cvtColor(imgAddW2, cv.COLOR_BGR2RGB))
    plt.subplot(1,3,3) , plt.title("(3) a=0.8, b=0.2"), plt.axis("off")
    plt.imshow(cv.cvtColor(imgAddW3, cv.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()


