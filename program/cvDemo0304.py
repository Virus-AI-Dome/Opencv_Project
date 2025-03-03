"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【0304】自定义色彩风格滤镜
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    # 读取原始图像
    img = cv.imread("../images/Fig0301.png", flags=1)  # 读取彩色图像
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]  # 图片的高度, 宽度

    plt.figure(figsize=(9, 6))
    plt.subplot(231), plt.axis('off'), plt.title("(1) Original")
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    from matplotlib import cm
    cmList = ["cm.copper", "cm.hot", "cm.YlOrRd", "cm.rainbow", "cm.prism"]
    for i in range(len(cmList)):
        cmMap = eval(cmList[i])(np.arange(256))
        print(cmMap)
        # RGB(matplotlib) -> BGR(OpenCV)
        lutC3 = np.zeros((1, 256, 3))  # BGR(OpenCV)
        lutC3[0, :, 0] = np.array(cmMap[:, 2] * 255).astype("uint8")  # B: cmHot[:, 2]
        lutC3[0, :, 1] = np.array(cmMap[:, 1] * 255).astype("uint8")  # G: cmHot[:, 1]
        lutC3[0, :, 2] = np.array(cmMap[:, 0] * 255).astype("uint8")  # R: cmHot[:, 0]

        cmLUTC3 = cv.LUT(img, lutC3).astype("uint8")
        print(img.shape, cmMap.shape, lutC3.shape)
        plt.subplot(2, 3, i + 2), plt.axis('off')
        plt.title("({}) {}".format(i + 2, cmList[i]))
        plt.imshow(cv.cvtColor(cmLUTC3, cv.COLOR_BGR2RGB))

    plt.tight_layout()
    plt.show()