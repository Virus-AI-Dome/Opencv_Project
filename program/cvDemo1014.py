"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【1014】拉普拉斯金字塔图像复原
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv.imread("../images/Fig0301.png", flags=1)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # 图像向下取样，构造高斯金字塔
    pyrG0 = img.copy()  # G0 (512,512)
    pyrG1 = cv.pyrDown(pyrG0)  # G1 (256,256)
    pyrG2 = cv.pyrDown(pyrG1)  # G2 (128,128)
    pyrG3 = cv.pyrDown(pyrG2)  # G3 (64,64)
    pyrG4 = cv.pyrDown(pyrG3)  # G4 (32,32)
    print("pyrG:", pyrG0.shape[:2], pyrG1.shape[:2], pyrG2.shape[:2], pyrG3.shape[:2], pyrG4.shape[:2])

    # 构造拉普拉斯金字塔，高斯金字塔的每一层减去其上一层图像的上采样
    pyrL0 = pyrG0 - cv.pyrUp(pyrG1)  # L0 (512,512)
    pyrL1 = pyrG1 - cv.pyrUp(pyrG2)  # L1 (256,256)
    pyrL2 = pyrG2 - cv.pyrUp(pyrG3)  # L2 (128,128)
    pyrL3 = pyrG3 - cv.pyrUp(pyrG4)  # L3 (64,64)
    print("pyrL:", pyrL0.shape[:2], pyrL1.shape[:2], pyrL2.shape[:2], pyrL3.shape[:2])

    # 向上采样恢复高分辨率图像
    rebuildG3 = pyrL3 + cv.pyrUp(pyrG4)
    rebuildG2 = pyrL2 + cv.pyrUp(rebuildG3)
    rebuildG1 = pyrL1 + cv.pyrUp(rebuildG2)
    rebuildG0 = pyrL0 + cv.pyrUp(rebuildG1)
    print("rebuild:", rebuildG0.shape[:2], rebuildG1.shape[:2], rebuildG2.shape[:2], rebuildG3.shape[:2])
    print("diff of rebuild：", np.mean(abs(rebuildG0 - img)))

    plt.figure(figsize=(10, 8))
    plt.subplot(341), plt.axis('off'), plt.title("GaussPyramid G0 "+str(pyrG0.shape[:2]))
    plt.imshow(cv.cvtColor(pyrG0, cv.COLOR_BGR2RGB))
    plt.subplot(342), plt.axis('off'), plt.title("G1 "+str(pyrG1.shape[:2]))
    plt.imshow(cv.cvtColor(pyrG1, cv.COLOR_BGR2RGB))
    plt.subplot(343), plt.axis('off'), plt.title("G2 "+str(pyrG2.shape[:2]))
    plt.imshow(cv.cvtColor(pyrG2, cv.COLOR_BGR2RGB))
    plt.subplot(344), plt.axis('off'), plt.title("G3 "+str(pyrG3.shape[:2]))
    plt.imshow(cv.cvtColor(pyrG3, cv.COLOR_BGR2RGB))
    plt.subplot(345), plt.axis('off'), plt.title("LaplacePyramid L0 " + str(pyrL0.shape[:2]))
    plt.imshow(cv.cvtColor(pyrL0, cv.COLOR_BGR2RGB))
    plt.subplot(346), plt.axis('off'), plt.title("L1 "+str(pyrL1.shape[:2]))
    plt.imshow(cv.cvtColor(pyrL1, cv.COLOR_BGR2RGB))
    plt.subplot(347), plt.axis('off'), plt.title("L2 "+str(pyrL2.shape[:2]))
    plt.imshow(cv.cvtColor(pyrL2, cv.COLOR_BGR2RGB))
    plt.subplot(348), plt.axis('off'), plt.title("L3 "+str(pyrL3.shape[:2]))
    plt.imshow(cv.cvtColor(pyrL3, cv.COLOR_BGR2RGB))
    plt.subplot(349), plt.axis('off'), plt.title("LaplaceRebuild R0 " + str(rebuildG0.shape[:2]))
    plt.imshow(cv.cvtColor(rebuildG0, cv.COLOR_BGR2RGB))
    plt.subplot(3, 4, 10), plt.axis('off'), plt.title("R1 "+str(rebuildG1.shape[:2]))
    plt.imshow(cv.cvtColor(rebuildG1, cv.COLOR_BGR2RGB))
    plt.subplot(3, 4, 11), plt.axis('off'), plt.title("R2 "+str(rebuildG2.shape[:2]))
    plt.imshow(cv.cvtColor(rebuildG2, cv.COLOR_BGR2RGB))
    plt.subplot(3, 4, 12), plt.axis('off'), plt.title("R3 "+str(rebuildG3.shape[:2]))
    plt.imshow(cv.cvtColor(rebuildG3, cv.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()
