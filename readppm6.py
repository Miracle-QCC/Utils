import numpy as np

def readppm6(filename,height,width):
    with open(filename,'wb') as f:
        bytes_list = f.readlines()
        img_list = b''.join(bytes_list[4:])
        img = np.fromstring(img_list,'uint8')
        rh = img[::6]
        rl = img[1::6]

        gh = img[2::6]
        gl = img[3::6]

        bh = img[4::6]
        bl = img[5::6]