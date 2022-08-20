import cv2
# import imageio
import numpy as np
# from PIL import Image
def ImageConvertPNG(filename):
    im = Image.open(filename)  # open ppm file

    newname = filename[:-4] + '.png'  # new name for png file
    im.save(newname)
def CV2ConvertPNG(filename):
    with open(filename,'rb') as f:
        tmp_list = []
        for i in range(4):
            data = f.readline()
            # print(data)
            tmp_list.append(data)
        width,height = map(int,tmp_list[2][:-1].split())
        depth = int(np.math.log(int(tmp_list[3][:-1]), 2)) + 1
        if depth > 8:
            img = np.zeros((height,width,3),dtype='uint16')
        else:
            img = np.zeros((height, width, 3), dtype='uint8')
        for y in range(height):
            for x in range(width):
                img[y, x, 0] = int.from_bytes(f.read(2), byteorder="big") # R
                img[y, x, 1] = int.from_bytes(f.read(2), byteorder="big") # G
                img[y, x, 2] = int.from_bytes(f.read(2), byteorder="big") # B
    # imageio.imwrite('outfile.png', img,)
    # img.tofile('outfile.png')
    scale = 255 / 4095
    img_scaled = img * scale

    cv2.imshow('aa',img_scaled)
    cv2.waitKey(0)
    # plt.imshow(img,  vmin=0, vmax=4096)
    # plt.show()

def MosaicPPM5(filename,out_path,height,width,depth):
    with open(filename, 'rb') as f:
        bytes_list = f.readlines()
        img_list = b''.join(bytes_list[4:])
        img = np.fromstring(img_list, 'uint8')
        rh = img[::6]
        rl = img[1::6]

        gh = img[2::6]
        gl = img[3::6]

        bh = img[4::6]
        bl = img[5::6]

        R = (rh * 256 + rl).reshape(height,width)
        G = (gh * 256 + gl).reshape(height,width)
        B = (bh * 256 + bl).reshape(height,width)
    array = np.transpose(np.stack((R, G, B)), (1, 2, 0)).astype('uint16')
    bayer = np.zeros(height*width,'uint16')
    for i in range(0,height,2):
        for j in range(0,width,2):
            bayer[i*width + j] = array[i][j][0]
            bayer[i*width + (j+1)] = array[i][j+1][1]
            bayer[(i+1)*width + j] = array[i+1][j][1]
            bayer[(i+1)*width + (j+1)] = array[i+1][j+1][2]
    # bayer = bayer.flatten()
    with open(out_path, 'w') as f:
        f.write("P5\n")
        f.write("# Created by CVITEK DPU Model\n")
        f.write("{} {}\n".format(width, height))
        f.write("{}\n".format((1 << depth) - 1))

    with open(out_path, 'ab') as f:
        # for i in range(0,height * width,4):
        #     rh = (int(bayer[i]) & 0x0000FF00) >> 8
        #     rl = (int(bayer[i]) & 0x000000FF) >> 0
        #
        #     grh = (int(bayer[i + 1]) & 0x0000FF00) >> 8
        #     grl = (int(bayer[i + 1]) & 0x000000FF) >> 0
        #
        #     gbh = (int(bayer[i + 2]) & 0x0000FF00) >> 8
        #     gbl = (int(bayer[i + 2]) & 0x000000FF) >> 0
        #
        #     bh = (int(bayer[i + 3]) & 0x0000FF00) >> 8
        #     bl = (int(bayer[i + 3]) & 0x000000FF) >> 0
        #
        #     f.write(chr(rh))
        #     f.write(chr(rl))
        #
        #     f.write(chr(grh))
        #     f.write(chr(grl))
        #
        #     f.write(chr(gbh))
        #     f.write(chr(gbl))
        #
        #     f.write(chr(bh))
        #     f.write(chr(bl))
        bayer.tofile(f)



def imread2PNG(filename):
    img = cv2.imread(filename,-1)
    scale = 255 / 4095
    img_scaled = img * scale
    img_scaled = img_scaled.astype('uint8')
    # cv2.imshow('a',img_scaled)
    # cv2.waitKey(0)
    cv2.imwrite('cfa_out_le_000.png',img_scaled)


def main(file_name, depth):
    if depth <= 8:
        ImageConvertPNG(file_name)
    else:
        CV2ConvertPNG(file_name)


if __name__ == '__main__':
    MosaicPPM5(r'E:\Program\DL\ISP\utils\cfa_out_le_000.ppm','cfa_mosaic.ppm',1080,1920,12)

    # CV2ConvertPNG('cfa_mosaic.ppm')
    # img = cv2.imread('cfa_out_le_000.ppm',-1)
    # cv2.imshow('a',img)
    # cv2.waitKey(0)