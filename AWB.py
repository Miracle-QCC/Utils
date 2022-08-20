import numpy as np
import PIL.Image as Image
import cv2

# 灰度世界方法
def AWB(img_file):
    img = Image.open(img_file).convert("RGB")
    img = np.array(img)
    R = img[:,:,0]
    G = img[:,:,1]
    B = img[:,:,2]

    r_gain = np.mean(R)
    g_gain = np.mean(G)
    b_gain = np.mean(B)

    R_ = R * (g_gain / r_gain)
    B_ = B * (g_gain / b_gain)

    img_ = np.stack((B_,G,R_))
    img_ = np.transpose(img_,(1,2,0)).astype('uint8')
    cv2.imwrite("data/wbg_raw.png",img_)

if __name__ == '__main__':
    AWB('data/only-pixel+1920X1080_RGGB_Linear_20211119183132_00_-color=3_-bits=12_-frame=1_-hdr=0_ISO=100_-modelout.png')