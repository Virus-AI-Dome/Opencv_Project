"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【0701】图像的反转变换
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    filepath ="../images/Lena.tif"
    img = cv.imread(filepath,flags=1)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    '''
    用 LUT 实现图像的 Gamma 校正（调整亮度和对比度），通常用于改善显示设备的图像呈现：
    gamma = 0.5
    transTable = np.array([((i/255)**gamma)*255 for i in range(0,256)]).astype("uint8")
    我们也可以使用 LUT 来增强图像的对比度。例如，通过简单的线性映射，放大较小的像素值差异：
    gamma = 0.5
    transTable = np.array([((i / 255) ** gamma) * 255 for i in range(0, 256)]).astype("uint8")
    '''
    transTable = np.array([(255-i)  for i in range(0, 256)]).astype("uint8")
    print(transTable)
    imgInv = cv.LUT(img, transTable)
    grayInv = cv.LUT(gray, transTable)



    plt.figure(figsize=(9, 3.5))
    plt.subplot(131), plt.title("(1) Original"), plt.axis('off')
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.subplot(132), plt.title("(2) Invert image"), plt.axis('off')
    plt.imshow(cv.cvtColor(imgInv, cv.COLOR_BGR2RGB))
    plt.subplot(133), plt.title("(3) Invert gray"), plt.axis('off')
    plt.imshow(grayInv, cmap='gray')
    plt.tight_layout()
    plt.show()
