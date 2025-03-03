"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【0108】多帧图像（动图）的读取和保存
import cv2 as cv
from matplotlib import pyplot as plt

if __name__ == '__main__':
    # 读取单幅图像，支持 bmp、jpg、png、tiff 等常用格式
    img1 = cv.imread("../images/FVid1.png")  # 读取彩色图像 FVid1.png
    img2 = cv.imread("../images/FVid2.png")  # 读取彩色图像 FVid2.png
    img3 = cv.imread("../images/FVid3.png")  # 读取彩色图像 FVid3.png
    img4 = cv.imread("../images/FVid4.png")  # 读取彩色图像 FVid4.png
    imgList = [img1, img2, img3, img4]  # 生成多帧图像列表

    # 保存多帧图像文件
    saveFile = "../images/imgList.tiff"  # 保存文件的路径
    ret = cv.imwritemulti(saveFile, imgList)
    if (ret):
        print("Image List Write Successed in {}".format(saveFile))
        print("len(imgList): ", len(imgList))  # imgList 是列表，只有长度没有形状

    # 读取多帧图像文件
    imgMulti = cv.imreadmulti("../images/imgList.tiff")  # 读取多帧图像文件
    print("len(imgList): ", len(imgList))  # imgList 是列表
    # 显示多帧图像文件
    for i in range(len(imgList)):
        print("\timgList[{}]: {}".format(i, imgList[i].shape))  # imgList[i] 是 Numpy 数组
        cv.imshow("imgList", imgList[i])  # 在窗口 imgList 逐帧显示
        cv.waitKey(100)
    cv.destroyAllWindows()



