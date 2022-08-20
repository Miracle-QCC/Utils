import cv2
import numpy as np


def ReadBayerPPM(single_filename):
    with open(single_filename, 'rb') as f:
        data = f.readlines()

        # width, height = map(int, data[2].split())
        # max_val = int(data[3])
        bytes_list = b''.join(data[4:])
        npdata = np.fromstring(bytes_list, 'uint8')
        # npdata = npdata.reshape(1080,1920)
        h = npdata[::2]
        l = npdata[1::2]
        r_data = (h * 256 + l).astype('uint16')
    return r_data


if __name__ == '__main__':
    unprocess = ReadBayerPPM('src_le_000.ppm').reshape(1080,1920).astype(np.float)
    process = ReadBayerPPM('cmodel_wb+out_le_000.ppm').reshape(1080,1920).astype(np.float)

    res = np.subtract(process,unprocess)
    abs_res = np.abs(res).astype('uint16')
    x = 1