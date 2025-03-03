"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【0407】添加英文文字与中文文字
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img1 = cv.imread("../images/Lena.tif")  # 读取彩色图像(BGR)
    img2 = img1.copy()

    # (1) cv.putText 添加非中文文字
    text = "Digital Image Processing, youcans@qq.com"  # 非中文文字
    fontList = [cv.FONT_HERSHEY_SIMPLEX,
                cv.FONT_HERSHEY_SIMPLEX,
                cv.FONT_HERSHEY_PLAIN,
                cv.FONT_HERSHEY_DUPLEX,
                cv.FONT_HERSHEY_COMPLEX,
                cv.FONT_HERSHEY_TRIPLEX,
                cv.FONT_HERSHEY_COMPLEX_SMALL,
                cv.FONT_HERSHEY_SCRIPT_SIMPLEX,
                cv.FONT_HERSHEY_SCRIPT_COMPLEX,
                cv.FONT_ITALIC]  # 字体设置
    fontScale = 0.8  # 字体缩放比例
    color = (255, 255, 255)  # 字体颜色
    for i in range(len(fontList)):
        pos = (10, 40*(i+1))  # 字符串左上角坐标 (x,y)
        cv.putText(img1, text, pos, fontList[i], fontScale, color)

    # (2) PIL 添加中文文字
    from PIL import Image, ImageDraw, ImageFont
    # if (isinstance(img2, np.ndarray)):  # 判断是否 OpenCV 图片类型
    imgPIL = Image.fromarray(cv.cvtColor(img2, cv.COLOR_BGR2RGB))
    string = '你好,opencv'
    pos= (50,20)
    color = (255, 255,0)
    textsize = 50
    drawPIL = ImageDraw.Draw(imgPIL)
    fontText = ImageFont.truetype("font/simsun.ttc", textsize, encoding="utf-8")
    drawPIL.text(pos, string, fontSize=textsize, color=color,font=fontText)
    imgText = cv.cvtColor(np.asarray(imgPIL), cv.COLOR_RGB2BGR)

    plt.figure(figsize = (9,3.5))
    plt.title("zhongwen")
    plt.axis('off')
    plt.imshow(cv.cvtColor(imgText, cv.COLOR_BGR2RGB))
    plt.show()