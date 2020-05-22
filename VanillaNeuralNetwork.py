# -*- coding: utf-8 -*-
"""
Created on Sat May 16 18:34:59 2020

@author: Tejo Nutalapati
"""

import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST




def relu(x):
    """returns max(0,x_i) element-wise for x"""
    x = np.array(x)

    return(x*(x>0))

def cross_entropy_loss(y,p,M):
    loss = 0
    observations = y.shape[0]
    for i in range(M):
        for o in range(observations):
            loss-= y*np.log(p)
            
    
    return(loss)

#TODO : functions to code or remove
def gradient():
    pass

def f():
    pass

def loss(y,y2):
    
    return((y-y2)**2)
    
def regularizer():
    pass

lamb = .2
    
def forwardprop(x,y,l,W,b):
    h = x    
    
    if W.shape[1] != x.shape[0]:
        return("W or x not proper shape")
    
    for k in range(l):
        a = b + W@h
        h = f(a)
        
    y_hat = h
    #cost
    J = loss(y,y_hat)+ lamb * regularizer(W,b)
    
    return(y_hat,J)


def backprop(y_hat,y,l,W,b):
    g = gradient(y_hat,y)
    f_prime = f
    
    for k in range(l-1,-1,-1):
        a = b + W@h
        g = g*f_prime(a)
        g_J_b = g + lamb*gradient(regularizer(W,b))        
        b = b +     g_J_b @ b        
        g_J_W = g*h + lamb*gradient(regularizer(W,b))
        W = W + g_J_W @W        
        g = W@g
        
    return(g)


mndata = MNIST('.')

train_images, train_labels = mndata.load_training()
train_images
# or
test_images, test_labels = mndata.load_testing()

# minibatch sampling

# X is an n x p design matrix
#X = train_images[np.random.randint(0,len(train_images),100)]

# activation function

## W must be a p x 
#W = np.array([[1,2],[2,1]])
#
#
##bias term
#b = np.array([1,2])
#
## affine transformation 
#h = relu(W @ x + b)

