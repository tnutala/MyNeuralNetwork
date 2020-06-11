# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 00:47:37 2020

@author: Tejo Nutalapati (@tnutala)
"""


import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST

mndata = MNIST('.')

train_images, train_labels = mndata.load_training()

# or
test_images, test_labels = mndata.load_testing()
