"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""


# 【0102】从网络地址读取图像
import cv2 as cv
import numpy as np

if __name__ == '__main__':
    import urllib.request as request
    response = request.urlopen\
        ("https://profile.csdnimg.cn/8/E/F/0_youcans")  # 指定的 url 地址
    imgUrl = cv.imdecode(np.array(bytearray(response.read()), dtype=np.uint8), -1)

    cv.imshow("imgUrl", imgUrl)  # 在窗口显示图像
    key = cv.waitKey(5000)  # 5000ms 后自动关闭
    cv.destroyAllWindows()



