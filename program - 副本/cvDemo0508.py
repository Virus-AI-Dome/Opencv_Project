"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【0507】模板匹配目标搜索
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    # 读取目标文件
    target = cv.imread("../images/Fig0943a.tif", flags=1)

    # 读取/选取匹配模板
    # templ = cv.imread("../images/Fig1105.tif", flags=1)
    print("Select a ROI and then press SPACE or ENTER button!\n")
    trackWindow = cv.selectROI(target, showCrosshair=True, fromCenter=False)
    x, y, w, h = trackWindow
    templ = target[y:y+h, x:x+w]  # 设置追踪区域
    hTpl, wTpl = templ.shape[:2]  # 模板图像的尺寸

    # 模板匹配
    method = cv.TM_CCOEFF_NORMED  # 设置匹配方法
    result = cv.matchTemplate(target, templ, method)

    # 查找最大值/最小值及其位置
    minVal, maxVal, minLoc, maxLoc = cv.minMaxLoc(result)
    if method==cv.TM_SQDIFF or method==cv.TM_SQDIFF_NORMED:
        ptTL = minLoc  # 方差匹配，查找最小值 (x,y)
        print("Minimum {:.4f} at {}: ".format(minVal, minLoc))
    else:  # 相关性匹配，查找最大值
        ptTL = maxLoc  # topleft
        print("Maximum {:.4f} at {}: ".format(maxVal, maxLoc))
    ptBR = (ptTL[0]+wTpl, ptTL[1]+hTpl)  # 右下点
    imgMatch1 = target.copy()
    cv.rectangle(imgMatch1, ptTL, ptBR, (0, 0, 255), 2)  # 绘制矩形

    threshold = 0.9
    loc = np.where(result >= threshold)
    imgMatch2 = target.copy()
    for pt in zip(*loc[::-1]):
        print("Maximum {:.4f} at {}. ".format(result[pt[1],pt[0]], pt))
        cv.rectangle(imgMatch2, pt, (pt[0]+wTpl, pt[1]+hTpl), (0, 0, 255), 2)

    plt.figure(figsize=(9, 6))
    # cv.rectangle(target, (x,y), (x+w,y+h), (255, 0, 255), 2)  # 标注原图
    # plt.axis('off'), plt.imshow(cv.cvtColor(target, cv.COLOR_BGR2RGB))
    plt.subplot(131), plt.title("Matching similarity")
    plt.axis('off'), plt.imshow(result, 'gray')
    plt.subplot(132), plt.title("Best matching")
    plt.axis('off'), plt.imshow(cv.cvtColor(imgMatch1, cv.COLOR_BGR2RGB))
    plt.subplot(133), plt.title("Approximated matching")
    plt.axis('off'), plt.imshow(cv.cvtColor(imgMatch2, cv.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()
