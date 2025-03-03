 """
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【0408】鼠标交互框选矩形区域 (ROI)
import cv2 as cv

if __name__ == '__main__':
    img = cv.imread("../images/Lena.tif")  # 读取彩色图像(BGR)
    # 鼠标框选矩形 ROI
    rect = cv.selectROI('ROI', img)  # 按下鼠标左键拖动，放开左键选中
    print("selectROI:", rect)  # 元组 (xmin, ymin, w, h)
    # 裁剪获取选择的矩形 ROI
    xmin, ymin, w, h = rect  # 矩形裁剪区域 (ymin:ymin+h, xmin:xmin+w) 的位置参数
    imgROI = img[ymin:ymin+h, xmin:xmin+w].copy()  # 切片获得裁剪后保留的图像区域
    # 显示选中的矩形 ROI
    cv.imshow("DemoRIO", imgROI)
    key = cv.waitKey()
    cv.destroyAllWindows()
