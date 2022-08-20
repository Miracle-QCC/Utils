import numpy as np
import cv2
import imageio
from PIL import Image

if __name__ == '__main__':
    # with open('cfa_mosaic.ppm', 'rb') as f:
    #     data = f.readlines()
    #
    #     # width, height = map(int, data[2].split())
    #     # max_val = int(data[3])
    #     bytes_list = b''.join(data[4:])
    #     npdata = np.fromstring(bytes_list, 'uint8')
    #     # npdata = npdata.reshape(1080,1920)
    #     h = npdata[::2]
    #     l = npdata[1::2]
    #     r_data = (h * 256 + l).astype('uint16')

    x= 1
    img = Image.open("data/0716.png").convert('RGB')
    img = np.array(img)
    ppm = np.zeros((2040,2040),'uint8')

    r_mask = np.zeros((2040,2040),'uint8')
    r_mask[::2,::2] = 1

    gr_mask = np.zeros((2040, 2040), 'uint8')
    gr_mask[::2, 1::2] = 1

    gb_mask = np.zeros((2040, 2040), 'uint8')
    gb_mask[1::2, ::2] = 1

    b_mask = np.zeros((2040, 2040), 'uint8')
    b_mask[1::2, 1::2] = 1

    ppm = r_mask * img[:,:,0] + gr_mask * img[:,:,1] + gb_mask * img[:,:,1] + b_mask * img[:,:,2]
    with open('data/0716_customize.ppm', 'w') as f:
        f.write("P5\n")
        f.write("# Created by CVITEK DPU Model\n")
        f.write("{} {}\n".format(2040, 2040))
        f.write("{}\n".format((1<<8) - 1))
    with open('data/0716_customize.ppm','ab') as f:
        ppm.tofile(f)
