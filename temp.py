import argparse
import os.path

import numpy as np
import time
from torchvision.transforms import Resize, CenterCrop, Compose
from PIL import Image

def timing1(func):
    def wrap(*args, **kwargs):
        time1 = time.time()
        res = func(*args, **kwargs)
        time2 = time.time()
        print(res, time2-time1)
    return wrap

def timing2():
    print(f'Time: {time.time()}')

@timing1
def add(a, b):
    return a + b

import torch
from torch import nn as nn

class Model(torch.nn.Module):
    def __int__(self):
        super(Model, self).__int__()
        self.layer1 = torch.nn.Conv2d(3, 64, kernel_size=1)
        self.layer2 = torch.nn.Linear(128)
        self.init_func = 'xavier'
        self.apply(self._init_weight)
        print(f'Hello world!')

    def _init_weight(self, m):
        if isinstance(m, nn.Linear):
            if self.init_func == 'xavier':
                nn.init.xavier_uniform_(m.weight)
            elif self.init_func == 'kaiming':
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            else:
                nn.init.xavier_uniform_(m.weight)

            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            if self.init_func == 'xavier':
                nn.init.xavier_uniform_(m.weight)
            elif self.init_func == 'kaiming':
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            else:
                nn.init.xavier_uniform_(m.weight)

            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        else:
            raise NotImplementedError

    def forward(self, x):
        return self.layer2(self.layer1(x))



parser = argparse.ArgumentParser()
parser.add_argument('--helps', action='store_false', default=True)
args = parser.parse_args()

if __name__ == '__main__':
    # rank = [1,5,10,128]
    # xl = lambda x : list('abc' + str(y) for y in x)
    # print(xl(rank))
    # path = 'test.jpeg'
    # trans = [Resize(256), CenterCrop(224)]
    # image = Image.open(path).convert('RGB')
    # print(image.size)
    # for trs in trans:
    #     image = trs(image)
    #     print(image.size)
    # model = Model()
    # print(model)
    # print(args.helps)
    a = ''
    c = 'a'
    b = 'b.pth'
    print(os.path.join(c, b), os.path.join(a, b))