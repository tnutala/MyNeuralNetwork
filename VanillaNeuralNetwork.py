# -*- coding: utf-8 -*-
"""
Created on Sat May 16 18:34:59 2020

@author: Tejo Nutalapati
"""

import numpy as np
import matplotlib.pyplot as plt



from mnist import MNIST

mndata = MNIST('.')

images, labels = mndata.load_training()
# or
images, labels = mndata.load_testing()



# activation function

W = np.array([[1,2],[2,1]])

h = g((W.transpose())@x +c)
