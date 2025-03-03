"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【0205】获取与修改像素值
import cv2 as cv

if __name__ == '__main__':
    filepath= "../images/Lena.tif"  # 读取文件的路径
    img = cv.imread(filepath, flags=1)  # flags=1 读取彩色图像(BGR)

    h, w = 20, 10  # 指定像素位置 (h,w)
    # (1) 直接访问数组元素，获取像素值
    pxBGR = img[h, w]  # 访问数组元素[h,w]，获取像素 (h,w) 的值
    print("(1) img[{},{}] = {}".format(h, w, img[h,w]))
    # (2) 直接访问数组元素，获取像素通道的值
    print("(2) img[{},{},ch]:".format(h,w))
    for i in range(3):
        print(img[h,w,i], end=' ')  # i=0,1,2 对应 B,G,R 通道
    # (3) img.item() 访问数组元素，获取像素通道的值
    print("\n(3) img.item({},{},ch):".format(h,w))
    for i in range(3):
        print(img.item(h,w,i), end=' ')  # i=0,1,2 对应 B,G,R 通道
    # (4) 修改像素值
    print("\n(4) old img[h,w] = {}".format(img[h,w]))
    img[h,w,:] = 255
    print("new img[h,w] = {}".format(img[h,w]))
