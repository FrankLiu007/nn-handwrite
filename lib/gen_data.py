#!/usr/bin/env python
#-*- coding: utf-8 -*-

'''

@author: yangfengguang

@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.

@contact: yangbisheng2009@gmail.com

@file: generate_data.py

@time: 2018/2/3 15:38

@desc:

'''

import numpy as np
# import matplotlib as plt
from sklearn import datasets

def gen_2dim_data():
    # Generate a dataset and plot it
    np.random.seed(0)
    X, y = datasets.make_moons(200, noise=0.20)
    # linux without UI can`t suppot this function
    # plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
    return X, y

if __name__ == '__main__'
    gen_2dim_data()