"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【1506】鼠标交互实现图割算法
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

drawing = False  # 绘图状态
mode = False  # 绘图模式

class GraphCut:
    def __init__(self, image):
        self.img = image
        self.imgRaw = img.copy()
        self.width = img.shape[0]
        self.height = img.shape[1]
        self.scale = 640 * self.width//self.height
        if self.width > 640:
            self.img = cv.resize(self.img, (640, self.scale), interpolation=cv.INTER_AREA)
        self.imgShow = self.img.copy()
        self.imgGauss = self.img.copy()
        self.imgGauss = cv.GaussianBlur(self.imgGauss, (3, 3), 0)
        self.lbUp = False
        self.rbUp = False
        self.lbDown = False
        self.rbDown = False
        self.mask = np.full(self.img.shape[:2], 2, dtype=np.uint8)
        self.firstChoose = True

def onMouseAction(event, x, y, flags, param):  # 鼠标交互
    global drawing, lastPoint, startPoint
    # 左键按下：开始画图
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        lastPoint = (x, y)
        startPoint = lastPoint
        param.lbDown = True
        print("Left button DOWN")
    elif event == cv.EVENT_RBUTTONDOWN:
        drawing = True
        lastPoint = (x, y)
        startPoint = lastPoint
        param.rbDown = True
        print("Right button DOWN")
    # 鼠标移动，画图
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing:
            if param.lbDown:
                cv.line(param.imgShow, lastPoint, (x, y), (0, 0, 255), 2, -1)
                cv.rectangle(param.mask, lastPoint, (x, y), 1, -1, 4)
            else:
                cv.line(param.imgShow, lastPoint, (x, y), (255, 0, 0), 2, -1)
                cv.rectangle(param.mask, lastPoint, (x, y), 0, -1, 4)
            lastPoint = (x, y)
    # 左键释放：结束画图
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        param.lbUp = True
        param.lbDown = False
        cv.line(param.imgShow, lastPoint, (x, y), (0, 0, 255), 2, -1)
        if param.firstChoose:
            param.firstChoose = False
        cv.rectangle(param.mask, lastPoint, (x, y), 1, -1, 4)
        print("Left button UP")
    elif event == cv.EVENT_RBUTTONUP:
        drawing = False
        param.rbUp = True
        param.rbDown = False
        cv.line(param.imgShow, lastPoint, (x, y), (255, 0, 0), 2, -1)
        if param.firstChoose:
            param.firstChoose = False
            param.mask = np.full(param.img.shape[:2], 3, dtype=np.uint8)
        cv.rectangle(param.mask, lastPoint, (x, y), 0, -1, 4)
        print("Right button UP")

if __name__ == '__main__':
    img = cv.imread("../images/Fig1502.png", flags=1)  # 读取彩色图像(BGR)
    graphCut = GraphCut(img)
    print("(1) 鼠标左键标记前景，鼠标右键标记背景")
    print("(2) 按 Esc 键退出，完成分割")

    # 定义鼠标的回调函数
    cv.namedWindow("image")
    cv.setMouseCallback("image", onMouseAction, graphCut)
    while (True):
        cv.imshow("image", graphCut.imgShow)
        if graphCut.lbUp or graphCut.rbUp:
            graphCut.lbUp = False
            graphCut.rbUp = False
            bgModel = np.zeros((1, 65), np.float64)
            fgModel = np.zeros((1, 65), np.float64)
            rect = (1, 1, graphCut.img.shape[1], graphCut.img.shape[0])
            mask = graphCut.mask
            graphCut.imgGauss = graphCut.img.copy()
            cv.grabCut(graphCut.imgGauss, mask, rect, bgModel, fgModel, 5, cv.GC_INIT_WITH_MASK)
            background = np.where((mask==2) | (mask==0), 0, 1).astype("uint8")  # 0 和 2 做背景
            graphCut.imgGauss = graphCut.imgGauss * background[:, :, np.newaxis]  # 使用掩模获取前景区域
            cv.imshow("result", graphCut.imgGauss)
        # 按下ESC键退出
        if cv.waitKey(20) == 27:
            break

    plt.figure(figsize=(9, 5.6))
    plt.subplot(221), plt.axis("off"), plt.title("(1) Original")
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))  # 显示 img(RGB)
    plt.subplot(222), plt.axis("off"), plt.title("(2) Mask image")
    plt.imshow(mask, "gray")
    plt.subplot(223), plt.axis("off"), plt.title("(3) Background")
    plt.imshow(background, "gray")
    plt.subplot(224), plt.axis("off"), plt.title("(4) Graph Cut")
    plt.imshow(cv.cvtColor(graphCut.imgGauss, cv.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()
