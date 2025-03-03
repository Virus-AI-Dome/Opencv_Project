"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【0208】LUT 查表实现颜色缩减
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':

    filepath = '../images/Lena.tif'
    gray = cv.imread(filepath, flags=0)
    h,w = gray.shape[:2]

    timeBegin = cv.getTickCount()
    imgGray32 = np.empty((w,h), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            imgGray32[i,j] = (gray[i,j]//8)*8    #返回为8的倍数
    timeEnd = cv.getTickCount()
    ## round 数据四舍五入
    print("Gray32  loop time: {} sec",format(round((timeBegin-timeEnd),4)))


    timeBegin = cv.getTickCount()
    table32 = np.array([(i//8)*8 for i in range(256)]).astype(np.uint8)
    print("label32", table32)
    gray32 = cv.LUT(gray, table32)
    timeEnd = cv.getTickCount()
    print("Grayscale reduction bu LUT :{} sec",format(round((timeBegin-timeEnd),4)))

    table8 = np.array([(i // 32) * 32 for i in range(256)]).astype(np.uint8)  # (256,)
    print("label8",table8)
    gray8 = cv.LUT(gray, table8)

    plt.figure(figsize=(9,6))
    plt.subplot(131)
    plt.title('gray-256')
    plt.axis('off')
    plt.imshow(gray, cmap='gray')

    plt.subplot(132)
    plt.title('gray-32')
    plt.axis('off')
    plt.imshow(gray32, cmap='gray')

    plt.subplot(133)
    plt.title('gray-8')
    plt.axis('off')
    plt.imshow(gray8, cmap='gray')

    plt.tight_layout()
    plt.show()



