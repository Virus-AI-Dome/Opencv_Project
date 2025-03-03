"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

#【1401】边缘检测之梯度算子
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv.imread("../images/Fig1401.png", flags=0)  # 灰度图像
    imgBlur = cv.blur(img, (3, 3))  # Blur 平滑

    # Laplacian 边缘算子
    kern_Laplacian_K1 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    kern_Laplacian_K2 = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    kern_Laplacian_K3 = np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]])  # 90 degree
    kern_Laplacian_K4 = np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]])  # -45 degree
    LaplacianK1 = cv.filter2D(imgBlur, -1, kern_Laplacian_K1)
    imgLaplacianK1 = cv.normalize(LaplacianK1, None, 0, 255, cv.NORM_MINMAX)
    LaplacianK2 = cv.filter2D(imgBlur, -1, kern_Laplacian_K2)
    imgLaplacianK2 = cv.normalize(LaplacianK2, None, 0, 255, cv.NORM_MINMAX)

    # Roberts 边缘算子
    kern_Roberts_x = np.array([[1, 0], [0, -1]])
    kern_Roberts_y = np.array([[0, -1], [1, 0]])
    imgRobertsX = cv.filter2D(img, -1, kern_Roberts_x)
    imgRobertsY = cv.filter2D(img, -1, kern_Roberts_y)
    imgRoberts = cv.convertScaleAbs(np.abs(imgRobertsX) + np.abs(imgRobertsY))

    # Prewitt 边缘算子
    kern_Prewitt_x = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    kern_Prewitt_y = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    imgPrewittX = cv.filter2D(img, -1, kern_Prewitt_x)
    imgPrewittY = cv.filter2D(img, -1, kern_Prewitt_y)
    imgPrewitt = cv.convertScaleAbs(np.abs(imgPrewittX) + np.abs(imgPrewittY))

    # Sobel 边缘算子
    kern_Sobel_x = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    kern_Sobel_y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    imgSobelX = cv.filter2D(img, -1, kern_Sobel_x)
    imgSobelY = cv.filter2D(img, -1, kern_Sobel_y)
    imgSobel = cv.convertScaleAbs(np.abs(imgSobelX) + np.abs(imgSobelY))

    plt.figure(figsize=(12, 8))
    plt.subplot(341), plt.title('Original')
    plt.axis('off'), plt.imshow(img, cmap='gray')
    plt.subplot(345), plt.title('Laplacian_K1')
    plt.axis('off'), plt.imshow(imgLaplacianK1, cmap='gray')
    plt.subplot(349), plt.title('Laplacian_K2')
    plt.axis('off'), plt.imshow(imgLaplacianK2, cmap='gray')
    plt.subplot(342), plt.title('Roberts')
    plt.axis('off'), plt.imshow(imgRoberts, cmap='gray')
    plt.subplot(346), plt.title('Roberts_X')
    plt.axis('off'), plt.imshow(imgRobertsX, cmap='gray')
    plt.subplot(3, 4, 10), plt.title('Roberts_Y')
    plt.axis('off'), plt.imshow(imgRobertsY, cmap='gray')
    plt.subplot(343), plt.title('Prewitt')
    plt.axis('off'), plt.imshow(imgPrewitt, cmap='gray')
    plt.subplot(347), plt.title('Prewitt_X')
    plt.axis('off'), plt.imshow(imgPrewittX, cmap='gray')
    plt.subplot(3, 4, 11), plt.title('Prewitt_Y')
    plt.axis('off'), plt.imshow(imgPrewittY, cmap='gray')
    plt.subplot(344), plt.title('Sobel')
    plt.imshow(imgSobel, cmap='gray'), plt.axis('off')
    plt.subplot(348), plt.title('Sobel_X')
    plt.axis('off'), plt.imshow(imgSobelX, cmap='gray')
    plt.subplot(3, 4, 12), plt.title('Sobel_Y')
    plt.axis('off'), plt.imshow(imgSobelY, cmap='gray')
    plt.tight_layout()
    plt.show()


