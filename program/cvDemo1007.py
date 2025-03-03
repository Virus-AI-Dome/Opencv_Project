"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【1007】空间滤波之自适应中值滤波器
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv.imread("../images/Fig1002.png", flags=0)  # 读取灰度图像
    hImg, wImg = img.shape[:2]
    print(hImg, wImg)

    # 边界填充
    smax = 7  # 允许最大窗口尺寸
    m, n = smax, smax  # 滤波器尺寸，m*n矩形邻域
    hPad, wPad = int((m-1)/2), int((n-1)/2)
    imgPad = cv.copyMakeBorder(img, hPad, hPad, wPad, wPad, cv.BORDER_REFLECT)

    imgMedianFilter = np.zeros(img.shape)  # 比较，中值滤波器
    imgAdaptMedFilter = np.zeros(img.shape)  # 自适应中值滤波器
    for h in range(hPad, hPad+hImg):
        for w in range(wPad, wPad+wImg):
            # (1) 中值滤波器 (Median filter)
            ksize = 3  # 固定邻域窗口尺寸
            kk = ksize//2  # 邻域半径
            win = imgPad[h-kk:h+kk+1, w-kk:w+kk+1]  # 邻域 Sxy, m*n
            imgMedianFilter[h-hPad, w-wPad] = np.median(win)

            # (2) 自适应中值滤波器 (Adaptive median filter)
            ksize = 3  # 自适应邻域窗口初值
            zxy = img[h-hPad, w-wPad]
            while True:
                k = ksize//2
                win = imgPad[h-k:h+k+1, w-k:w+k+1]  # 邻域 Sxy(ksize)
                zmin, zmed, zmax = np.min(win), np.median(win), np.max(win)
                if zmin < zmed < zmax:  # Zmed 不是噪声
                    if zmin < zxy < zmax:
                        imgAdaptMedFilter[h-hPad, w-wPad] = zxy
                    else:
                        imgAdaptMedFilter[h-hPad, w-wPad] = zmed
                    break
                else:
                    if ksize >= smax:  # 达到最大窗口
                        imgAdaptMedFilter[h-hPad, w-wPad] = zmed
                        break
                    else:  # 未达到最大窗口
                        ksize = ksize + 2  # 增大窗口尺寸

    plt.figure(figsize=(9, 3.5))
    plt.subplot(131), plt.axis('off'), plt.title("(1) Original")
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.subplot(132), plt.axis('off'), plt.title("(2) Median filter")
    plt.imshow(imgMedianFilter, cmap='gray', vmin=0, vmax=255)
    plt.subplot(133), plt.axis('off'), plt.title("(3) Adaptive median filter")
    plt.imshow(imgAdaptMedFilter, cmap='gray', vmin=0, vmax=255)
    plt.tight_layout()
    plt.show()
