#!/usr/bin/env python
#-*- coding: utf-8 -*-

import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import os
import cPickle

import numpy as np
# import matplotlib as plt
from sklearn import datasets

home_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
bin_dir = os.path.join(home_dir, 'bin')
model_dir = os.path.join(home_dir, 'model')
conf_dir = os.path.join(home_dir, 'conf')
data_dir = os.path.join(home_dir, 'data')
lib_dir = os.path.join(home_dir, 'lib')
log_dir = os.path.join(home_dir, 'log')


def gen_2dim_data():
    # Generate a dataset and plot it
    np.random.seed(0)
    X, y = datasets.make_moons(200, noise=0.20)
    # linux without UI can`t suppot this function
    # plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
    cPickle.dump(X, open(os.path.join(data_dir, 'moon.X.pkl'), 'wb'))
    cPickle.dump(y, open(os.path.join(data_dir, 'moon.y.pkl'), 'wb'))
    f = open(os.path.join(data_dir, 'moon.data'), 'wb')
    for o in zip(y, X):
        f.write(str(o)+'\n')
    
    return X, y

if __name__ == '__main__':
    gen_2dim_data()
