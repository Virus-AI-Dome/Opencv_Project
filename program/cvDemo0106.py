"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【0106】视频文件的读取、播放和保存
import cv2 as cv

if __name__ == '__main__':
    # 创建视频读取/捕获对象
    vedioRead = "../images/vedioDemo1.mov"  # 读取视频文件的路径
    capRead = cv.VideoCapture(vedioRead)  # 实例化 VideoCapture 类

    # 设置写入视频图像的高，宽，帧速率和总帧数
    width = int(capRead.get(cv.CAP_PROP_FRAME_WIDTH))  # 960
    height = int(capRead.get(cv.CAP_PROP_FRAME_HEIGHT))  # 540
    fps = round(capRead.get(cv.CAP_PROP_FPS))  # 30
    frameCount = int(capRead.get(cv.CAP_PROP_FRAME_COUNT))  # 1826
    print(height, width, fps, frameCount)

    # 创建写入视频对象
    # fourcc = cv.VideoWriter_fourcc('X', 'V', 'I', 'D')  # 编码器设置 XVID
    fourcc = cv.VideoWriter_fourcc(*'XVID')  # 'X','V','I','D' 简写为 *'XVID'
    vedioWrite = "../images/vedioSave1.avi"  # 写入视频文件的路径
    capWrite = cv.VideoWriter(vedioWrite, fourcc, fps, (width, height))

    # 读取视频文件，抽帧写入视频文件
    frameNum = 0  # 视频帧数初值
    timef = 30  # 设置抽帧间隔
    while capRead.isOpened():  # 检查视频捕获是否成功
        ret, frame = capRead.read()  # 读取下一帧视频图像
        if ret is True:
            frameNum += 1  # 读取视频的帧数
            cv.imshow(vedioRead, frame)  # 播放视频图像
            if (frameNum % timef == 0):  # 判断抽帧条件
                capWrite.write(frame)  # 将当前帧写入视频文件
            if cv.waitKey(1) & 0xFF == ord('q'):  # 按 'q' 退出
                break
        else:
            print("Can't receive frame at frameNum {}.".format(frameNum))
            break

    capRead.release()  # 关闭读取视频文件
    capWrite.release()  # 关闭视频写入对象
    cv.destroyAllWindows()  # 关闭显示窗口

    # cap = cv.VideoCapture(vedioWrite)  # 实例化 VideoCapture 类
    # frameWrite = int(cap.get(cv.CAP_PROP_FRAME_COUNT))  # 1826
    # cap.release()  # 关闭读取视频文件
    # print("Read frame:", frameNum)  # 读取视频帧数
    # print("Write frame:", frameWrite)  # 写入视频帧数