"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【0206】图像的马赛克处理
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':

    filepath = '../images/lena.tif'
    img = cv.imread(filepath)
    roi = cv.selectROI('Select ROI', img, False)
    x, y, wRoi, hRoi = roi
    imgRoi = img[y:y+hRoi, x:x+wRoi]

    plt.figure(figsize=(9,6))
    plt.subplot(2,3,1)
    plt.title('Original Image')
    plt.axis('off')
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))

    plt.subplot(2,3,2)
    plt.title('ROI')
    plt.axis('off')
    plt.imshow(cv.cvtColor(imgRoi, cv.COLOR_BGR2RGB))

    mosaic = np.zeros(imgRoi.shape, dtype='uint8')
    ksize = [5,10,20]
    for i in range(3):
        k = ksize[i]
        for h in range(0, hRoi, k):
            for w in range(0, wRoi, k):
                mosaic[h:h+k,w:w+k] = imgRoi[h,w]
        imgMosaic = img.copy()
        imgMosaic[y:y+hRoi,x:x+wRoi] = mosaic
        plt.subplot(2,3,i+4)
        plt.title('({}) Coding Image size({})'.format(i+1,k))
        plt.axis('off')
        plt.imshow(cv.cvtColor(imgMosaic, cv.COLOR_BGR2RGB))

    plt.subplot(2,3,3)
    plt.title('(3) Mosaic ')
    plt.axis('off')
    plt.imshow(cv.cvtColor(imgRoi, cv.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()
