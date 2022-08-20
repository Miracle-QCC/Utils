import numpy as np
import cv2


def ReverseGamma(img, gamma = 2.2,c=1.0):
    out = c * np.power(img / 255.0, gamma) * c * 255.0

    return out.astype('uint8')



def main(img_file):
    img = cv2.imread(img_file)
    r_gamma_img = ReverseGamma(img)

    return  r_gamma_img

if __name__ == '__main__':
    main('0002.png')