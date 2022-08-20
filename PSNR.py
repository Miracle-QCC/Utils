import sys
import cv2
import numpy as np
import math

from PIL import Image


def psnr(target, ref):
    # 将图像格式转为float64
    target_data = np.array(target, dtype=np.float64)
    ref_data = np.array(ref, dtype=np.float64)
    # 直接相减，求差值
    diff = ref_data - target_data
    # 按第三个通道顺序把三维矩阵拉平
    diff = diff.flatten('C')
    # 计算MSE值
    rmse = math.sqrt(np.mean(diff ** 2.))
    # 精度
    eps = np.finfo(np.float64).eps
    if (rmse == 0):
        rmse = eps
    return 20 * math.log10(1 / rmse)

if __name__ == '__main__':
    img1 = Image.open('data/10blocks_2560X1440_GRBG_Linear_20220805160539_00_-color=2_-bits=12_-frame=1_-hdr=0_ISO=130_-modelout.png').convert('RGB')
    img2 = Image.open('data/2560X1440_GRBG_Linear_20220805160539_00_-color=2_-bits=12_-frame=1_-hdr=0_ISO=130_.bmp').convert('RGB')
    # 计算单通道图像
    img1 = np.array(img1) / 255.0
    img2 = np.array(img2) / 255.0
    print(psnr(img1,img2))

