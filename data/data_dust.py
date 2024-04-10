import os
from PIL import Image
import cv2, math
import numpy as np

# 源目录
myPath = 'F:\DustGAN\datasets\original'
# 输出目录
outPath = 'F:\DustGAN\datasets\dust'


def processImage(filesource, destsource, name, imgtype):

    imgtype = 'jpeg' if imgtype == '.jpg' else 'png'
    # 打开图片
    img = cv2.imread(name)
    img_f = img / 255.0
    (row, col, chs) = img.shape;

    A = 0.7  # 亮度
    beta = 0.025  # 粉尘的浓度
    size = math.sqrt(max(row, col))  # 粉尘尺寸
    center = (row // 1.5, col // 2)  # 粉尘化中心
    for j in range(row):
        for l in range(col):
            d = -0.04 * math.sqrt((j - center[0]) ** 2 + (l - center[1]) ** 2) + size
            td = math.exp(-beta * d)
            img_f[j][l][:] = img_f[j][l][:] * td + A * (1 - td)
    cv2.imwrite(destsource + name, img_f * 255)


def run():
    # 切换到源目录，遍历目录下所有图片
    os.chdir(myPath)
    file_list = sorted(os.listdir(os.getcwd()), key=lambda x: int(x.split(".")[0]))
    for i in file_list:
        # 检查后缀
        postfix = os.path.splitext(i)[1]
        print(postfix, i)
        if postfix == '.jpg' or postfix == '.png':
            processImage(myPath, outPath, i, postfix)


if __name__ == '__main__':
    run()
