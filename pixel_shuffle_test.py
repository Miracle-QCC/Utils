import torch
import torch.nn as nn


if __name__ == '__main__':
    a = torch.randn(1,9,1,1)
    r = nn.PixelShuffle(3)
    b = r(a)
    print(a)