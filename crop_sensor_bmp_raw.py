import os
import os.path as osp
import time
from glob import glob
import ctypes
import cv2
import numpy as np
import torch
import tqdm
from PIL import Image
import argparse
from multiprocessing import Pool
from scipy.io import loadmat, savemat


def main():
    parser = argparse.ArgumentParser(description='A multi-thread tool to crop sub images')
    parser.add_argument('--src_path', type=str, default=r'E:\Program\DL\ISP\utils\raw',
                        help='path to original mat folder')
    parser.add_argument('--save_path', type=str, default=r'E:\Program\DL\ISP\utils\new_crop_raw',
                        help='path to output folder')
    args = parser.parse_args()
    args.save_path = osp.join(os.getcwd(), args.save_path)

    os.makedirs(args.save_path, exist_ok=True)

    n_thread = 32
    crop_sz = 256
    stride = 128
    thres_sz = 128  # keep the regions in the last row/column whose thres_sz is over 256.
    ext = 'bmp'
    r_ext = 'raw'
    bmp_list = sorted(
        glob(osp.join(os.getcwd(), args.src_path, '*' + ext))
    )
    raw_list = sorted(
        glob(osp.join(os.getcwd(), args.src_path, '*' + r_ext))
    )

    pool = Pool(n_thread)
    # worker_cropraw_bmp(raw_list[0], bmp_list[0], args.save_path, crop_sz, stride, thres_sz)

    for i in tqdm.tqdm(range(len(raw_list))):
        print('processing {}/{}'.format(i+1, len(bmp_list)))
        pool.apply_async(worker_cropraw_bmp, args=(raw_list[i],bmp_list[i], args.save_path, crop_sz, stride, thres_sz))
    pool.close()
    pool.join()

    print('All subprocesses done.')


# 只负责将raw图转换为4通道，然后与bmp图一起裁剪成指定大小，最后存为mat中
# 'raw': patch_raw
# 'gt':patch_bmp
# 针对GRBG格式的raw图
def worker_cropraw_bmp(raw_path, bmp_path, save_dir, crop_sz, stride, thres_sz):
    raw_name = osp.basename(raw_path)
    bmp_name = osp.basename(bmp_path)
    bmp = Image.open(bmp_path).convert('RGB')
    bmp = np.array(bmp)

    h, w,c = bmp.shape
    raw = np.fromfile(raw_path,'uint16').reshape(h,w)
    raw_mat = np.zeros((h,w,4),'uint16')

    ##### GRBG  -> RGGB
    # Gr
    raw_mat[::2,1::2,1] = raw[::2,::2]

    # R
    raw_mat[::2,::2,0] = raw[::2,1::2]

    # B
    raw_mat[1::2,1::2,3] = raw[1::2,::2]

    #Gb
    raw_mat[1::2,::2,2] = raw[1::2,1::2]


    h_space = np.arange(0, h - crop_sz + 1, stride)
    if h - (h_space[-1] + crop_sz) > thres_sz:
        h_space = np.append(h_space, h - crop_sz)
    w_space = np.arange(0, w - crop_sz + 1, stride)
    if w - (w_space[-1] + crop_sz) > thres_sz:
        w_space = np.append(w_space, w - crop_sz)

    index = 0
    for x in h_space:
        for y in w_space:
            index += 1
            patch_bmp_name = bmp_name.replace('.bmp', '_s{:05d}.bmp'.format(index))
            patch_raw_name = raw_name.replace('.raw', '_s{:05d}.mat'.format(index))
            patch_bmp = bmp[x:x + crop_sz, y:y + crop_sz, :]
            patch_raw = raw_mat[x:x + crop_sz, y:y + crop_sz,:]

            # patch_bmp.save(os.path.join(save_dir,patch_bmp_name))
            savemat(os.path.join(save_dir,patch_raw_name), {'raw': patch_raw,'gt':patch_bmp},)
            # sotools.crop_raw(raw_, x,x + crop_sz, y, y + crop_sz,os.path.join(save_dir,patch_raw_name))

if __name__ == '__main__':

    # sotools = ctypes.cdll.LoadLibrary('tools.so')
    # main()
    data = np.fromfile(r'E:\Program\DL\ISP\utils\raw\0001.raw','uint16')
    data_ = data
    with open(r'E:\Program\DL\ISP\utils\raw\0001.ppm','w') as f:

            f.write("P5\n")
            f.write("# Created by CVITEK DPU Model\n")
            f.write("{} {}\n".format(2560, 1440))
            f.write("{}\n".format((1 << 12) - 1))
    with open(r'E:\Program\DL\ISP\utils\raw\0001.ppm', 'ab') as f:
            data_.tofile(f)
