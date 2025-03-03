"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【0608】图像重映射实现动画播放效果
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def updateMapXY(mapx, mapy, s):
    height, width = img.shape[:2]
    scale = 0.1 + 0.9 * s / 100  # 0.1->1.0
    padx = 0.5 * width * (1 - scale)  # 左右填充
    pady = 0.5 * height * (1 - scale)  # 上下填充
    mapx = np.array([[((j-padx)/scale) for j in range(width)] for i in range(height)], np.float32)
    mapy = np.array([[((i-pady)/scale) for j in range(width)] for i in range(height)], np.float32)
    return mapx, mapy

if __name__ == '__main__':
    img = cv.imread("../images/Fig0301.png")  # 读取彩色图像(BGR)
    height, width = img.shape[:2]  # (512, 512, 3)
    mapx = np.zeros(img.shape[:2], np.float32)
    mapy = np.zeros(img.shape[:2], np.float32)
    borderColor = img[-1, -1, :].tolist()  # 填充背景颜色
    dst = np.zeros(img.shape, np.uint8)
    print(img.shape, dst.shape)

    for s in range(100):
        key = 0xFF & cv.waitKey(10)  # 按 ESC 退出
        if key == 27:  # esc to exit
            break
        mapx, mapy = updateMapXY(mapx, mapy, s)
        dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR, borderValue=borderColor)
        cv.imshow("RemapWin", dst)
    cv.destroyAllWindows()  # 图像窗口

    plt.figure(figsize=(9, 3.5))
    sList = [20, 50, 80]
    for i in range(len(sList)):
        mapx, mapy = updateMapXY(mapx, mapy, s=sList[i])
        dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR, borderValue=borderColor)
        plt.subplot(1,3,i+1), plt.title("({}) Dynamic (t={})".format(i+1, sList[i]))
        plt.axis('off'), plt.imshow(cv.cvtColor(dst, cv.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()
