"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【1707】特征检测之最大稳定极值区域（MSER）
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def NonMaxSuppression(boxes, thresh=0.5):
    x1, y1 = boxes[:,0], boxes[:,1]
    x2, y2 = boxes[:,0]+boxes[:,2], boxes[:,1]+boxes[:,3]
    area = boxes[:,2] * boxes[:,3]  # 计算面积
    # 删除重复的矩形框
    pick = []
    idxs = np.argsort(y2)  # 返回的是右下角坐标从小到大的索引值
    while len(idxs) > 0:
        last = len(idxs) - 1  # 将最右下方的框放入pick 数组
        i = idxs[last]
        pick.append(i)
        # 剩下框中最大的坐标(x1Max,y1Max)和最小的坐标(x2Min,y2Min)
        x1Max = np.maximum(x1[i], x1[idxs[:last]])
        y1Max = np.maximum(y1[i], y1[idxs[:last]])
        x2Min = np.minimum(x2[i], x2[idxs[:last]])
        y2Min = np.minimum(y2[i], y2[idxs[:last]])
        # 重叠面积的占比
        w = np.maximum(0, x2Min-x1Max+1)
        h = np.maximum(0, y2Min-y1Max+1)
        overlap = (w * h) / area[idxs[:last]]
        # 根据重叠占比的阈值删除重复的矩形框
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > thresh)[0])))
    return boxes[pick]  # x, y, w, h

if __name__ == '__main__':
    img = cv.imread("../images/Fig1702.png", flags=1)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]

    # 创建 MSER 对象，检测 MSER 区域
    mser = cv.MSER_create(min_area=60, max_area=300)  # 实例化 MSER
    # mser = cv.MSER_create(_min_area=500, _max_area=20000)  # 部分版本的格式不同
    regions, boxes = mser.detectRegions(gray)  # 检测并返回找到的 MSER
    lenMSER = len(regions)
    print("Number of detected MSER: ", lenMSER)

    imgMser1 = img.copy()
    imgMser2 = img.copy()
    for i in range(lenMSER):
        # 绘制 MSER 凸壳
        points = regions[i].reshape(-1, 1, 2)  # (k,2) -> (k,1,2)
        hulls = cv.convexHull(points)
        cv.polylines(imgMser1, [hulls], 1, (0,255,0), 2)  # 绘制凸壳 (x,y)
        #　绘制 MSER 矩形框
        x, y, w, h = boxes[i]  # 区域的垂直矩形边界框
        cv.rectangle(imgMser2, (x,y), (x+w,y+h), (0,255,0), 2)

    # 非最大值抑制 (NMS)
    imgMser3 = img.copy()
    nmsBoxes = NonMaxSuppression(boxes, 0.6)
    lenNMS = len(nmsBoxes)
    print("Number of NMS-MSER: ", lenNMS)
    for i in range(lenNMS):
        #　绘制 NMS-MSER 矩形框
        x, y, w, h = nmsBoxes[i]  # NMS 矩形框
        cv.rectangle(imgMser3, (x,y), (x+w,y+h), (0,255,0), 2)

    plt.figure(figsize=(9, 3.2))
    plt.subplot(131), plt.title("(1) MSER regions")
    plt.axis('off'), plt.imshow(cv.cvtColor(imgMser1, cv.COLOR_BGR2RGB))
    plt.subplot(132), plt.title("(2) MSER boxes")
    plt.axis('off'), plt.imshow(cv.cvtColor(imgMser2, cv.COLOR_BGR2RGB))
    plt.subplot(133), plt.title("(3) NMS-MSER boxes")
    plt.axis('off'), plt.imshow(cv.cvtColor(imgMser3, cv.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()
