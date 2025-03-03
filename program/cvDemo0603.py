"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【0603】图像的旋转
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv.imread("../images/Fig0301.png")
    height, width = img.shape[:2]

    # (1) 以原点为中心旋转
    x0, y0 = 0, 0
    theta, scale = 30, 10
    MARO = cv.getRotationMatrix2D((x0, y0), theta, scale)
    imgRot1 = cv.warpAffine(img, MARO, (width, height))
    # (2) 以任意点为中心旋转
    x0, y0 = width//2, height//2
    angle = theta * np.pi/180

    wRot, hRot = int(width * np.cos(angle) + height * np.sin(angle)), int(height * np.cos(angle) + width * np.sin(angle) )
    scale = width / wRot

    MAR1 = cv.getRotationMatrix2D((x0, y0), theta, 0.6)
    MAR2 = cv.getRotationMatrix2D((x0, y0), theta, scale)

    imgRot2 = cv.warpAffine(img, MAR1, (width, height), borderValue=(255, 255, 255))
    imgRot3 = cv.warpAffine(img, MAR2, (width, height))

    print(img.shape, imgRot2.shape, imgRot3.shape,scale)

    imgRot90 = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
    imgRot180 = cv.rotate(img, cv.ROTATE_180)
    imgRot270 = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)
    plt.figure(figsize=(9, 6))
    plt.subplot(231), plt.title("(1) Rotate around the origin"), plt.axis('off')
    plt.imshow(cv.cvtColor(imgRot1, cv.COLOR_BGR2RGB))
    plt.subplot(232), plt.title("(2) Rotate around the center"), plt.axis('off')
    plt.imshow(cv.cvtColor(imgRot2, cv.COLOR_BGR2RGB))
    plt.subplot(233), plt.title("(3) Rotate and resize"), plt.axis('off')
    plt.imshow(cv.cvtColor(imgRot3, cv.COLOR_BGR2RGB))
    plt.subplot(234), plt.title("(4) Rotate 90 degrees"), plt.axis('off')
    plt.imshow(cv.cvtColor(imgRot90, cv.COLOR_BGR2RGB))
    plt.subplot(235), plt.title("(5) Rotate 180 degrees"), plt.axis('off')
    plt.imshow(cv.cvtColor(imgRot180, cv.COLOR_BGR2RGB))
    plt.subplot(236), plt.title("(6) Rotate 270 degrees"), plt.axis('off')
    plt.imshow(cv.cvtColor(imgRot270, cv.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()





