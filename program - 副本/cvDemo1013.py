"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【1013】高斯图像金字塔 (Gaussian pyramid)
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv.imread("../images/Fig0301.png", flags=1)
    print(img.shape)

    # 图像向下取样
    pyrG0 = img.copy()  # G0 (512,512)
    pyrG1 = cv.pyrDown(pyrG0)  # G1 (256,256)
    pyrG2 = cv.pyrDown(pyrG1)  # G2 (128,128)
    pyrG3 = cv.pyrDown(pyrG2)  # G3 (64,64)
    print(pyrG0.shape, pyrG1.shape, pyrG2.shape, pyrG3.shape)

    # 图像向上取样
    pyrU3 = pyrG3.copy()  # U3 (64,64)
    pyrU2 = cv.pyrUp(pyrU3)  # U2 (128,128)
    pyrU1 = cv.pyrUp(pyrU2)  # U1 (256,256)
    pyrU0 = cv.pyrUp(pyrU1)  # U0 (512,512)
    print(pyrU3.shape, pyrU2.shape, pyrU1.shape, pyrU0.shape)

    plt.figure(figsize=(9, 5))
    plt.subplot(241), plt.axis('off'), plt.title("G0 "+str(pyrG0.shape[:2]))
    plt.imshow(cv.cvtColor(pyrG0, cv.COLOR_BGR2RGB))
    plt.subplot(242), plt.axis('off'), plt.title("->G1 "+str(pyrG1.shape[:2]))
    down1 = np.ones_like(img, dtype=np.uint8)*128
    down1[:pyrG1.shape[0], :pyrG1.shape[1], :] = pyrG1
    plt.imshow(cv.cvtColor(down1, cv.COLOR_BGR2RGB))
    plt.subplot(243), plt.axis('off'), plt.title("->G2 "+str(pyrG2.shape[:2]))
    down2 = np.ones_like(img, dtype=np.uint8)*128
    down2[:pyrG2.shape[0], :pyrG2.shape[1], :] = pyrG2
    plt.imshow(cv.cvtColor(down2, cv.COLOR_BGR2RGB))
    plt.subplot(244), plt.axis('off'), plt.title("->G3 "+str(pyrG3.shape[:2]))
    down3 = np.ones_like(img, dtype=np.uint8)*128
    down3[:pyrG3.shape[0], :pyrG3.shape[1], :] = pyrG3
    plt.imshow(cv.cvtColor(down3, cv.COLOR_BGR2RGB))
    plt.subplot(245), plt.axis('off'), plt.title("U0 "+str(pyrU0.shape[:2]))
    up0 = np.ones_like(img, dtype=np.uint8)*128
    up0[:pyrU0.shape[0], :pyrU0.shape[1], :] = pyrU0
    plt.imshow(cv.cvtColor(up0, cv.COLOR_BGR2RGB))
    plt.subplot(246), plt.axis('off'), plt.title("<-U1 " + str(pyrU1.shape[:2]))
    up1 = np.ones_like(img, dtype=np.uint8)*128
    up1[:pyrU1.shape[0], :pyrU1.shape[1], :] = pyrU1
    plt.imshow(cv.cvtColor(up1, cv.COLOR_BGR2RGB))
    plt.subplot(247), plt.axis('off'), plt.title("<-U2 " + str(pyrU2.shape[:2]))
    up2 = np.ones_like(img, dtype=np.uint8)*128
    up2[:pyrU2.shape[0], :pyrU2.shape[1], :] = pyrU2
    plt.imshow(cv.cvtColor(up2, cv.COLOR_BGR2RGB))
    plt.subplot(248), plt.axis('off'), plt.title("<-U3 " + str(pyrU3.shape[:2]))
    up3 = np.ones_like(img, dtype=np.uint8)*128
    up3[:pyrU3.shape[0], :pyrU3.shape[1], :] = pyrU3
    plt.imshow(cv.cvtColor(up3, cv.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()
