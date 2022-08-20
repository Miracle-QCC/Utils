import os

import numpy as np
import cv2
from scipy.io import loadmat
from tqdm import tqdm
import pandas as pd

# def get(file_path):
#     data = []
#     for file in tqdm(os.listdir(file_path)):
#         matfile = loadmat(os.path.join(file_path,file))
#         rggb = np.asarray(matfile['ps4k'])
#         data.append([rggb.mean() , rggb.std()])
#
#     df = pd.DataFrame(data,columns=['mean','std'],dtype=float)
#
#     df.to_csv('pixelshift.csv')
import os

import numpy as np
from PIL import Image

from tqdm import tqdm
import pandas as pd


def get(file):

    img = Image.open(file)
    rgb = np.array(img)
    bayer = np.zeros((rgb.shape[:2]), dtype='uint8')
    for i in range(0, rgb.shape[0], 2):
        for j in range(0, rgb.shape[1], 2):
            bayer[i][j] = rgb[i][j][0]
            bayer[i][j + 1] = rgb[i][j + 1][1]
            bayer[i + 1][j] = rgb[i + 1][j][1]
            bayer[i + 1][j + 1] = rgb[i + 1][j + 1][2]

    x = 1

def get_ppm(file):
    img = cv2.imread(file, -1)
    img = img / 4095.0
    return img.mean(),img.std()

if __name__ == '__main__':
    print(get_ppm('wb_out_le_000.ppm'))
