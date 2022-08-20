import numpy as np
import ctypes

if __name__ == '__main__':
    fun = ctypes.cdll.LoadLibrary('tools.so')
    data = np.fromfile('0001.raw',dtype='uint16').reshape(1440,2560)
    data_ = data.astype(np.int)
    arr = data_.ctypes.data_as(ctypes.c_char_p)
    fun.crop_raw(arr,0,2,0,2,"abc\\a")


