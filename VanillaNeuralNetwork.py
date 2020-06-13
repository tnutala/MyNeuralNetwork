# -*- coding: utf-8 -*-
"""
Created on Sat May 16 18:34:59 2020

@author: Tejo Nutalapati
"""

import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST


mndata = MNIST('.')

train_images, train_labels = mndata.load_training()

# or
test_images, test_labels = mndata.load_testing()



"""
Training Set using stochastic gradient descent. 
It should achieve 97-98% accuracy on the Test Set
"""


class MyNeuralNet():
    # TODO: what does 'super()' do?
    super().__init__
    
    def __init__(self,num_layers=2,num_hidden=[2,2],learning_rate=.01):
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.lr = learning_rate
        
        
    def relu(x):
        """returns max(0,x_i) element-wise for x"""
        x = np.array(x)
        
        return(x*(x>0))

    def forward(self,x):
        
    
        
        # initializing the weights and bias terms        
        w = [0]*(self.num_layers)
        b = [0]*(self.num_layers)
        #actually the transposed weight

        for k in range(0,self.num_layers):
            
            # TODO: the weight shapes need to be adjusted if each hidden layer has different
            # number of nodes
            W_k = np.ones((self.num_hidden[k],w[k-1].shape[0])) 
            b_k = np.ones((self.num_hidden,1))     
            w[k] = W_k
            b[k] = b_k
        
        # initializing the hidden layers
        h = [0]*(self.num_layers+1)
        
        activation_function = relu

        h[0] = x

        # forward pass
        for k in range(1,self.num_layers+1):
            a_k = w[k] @ h[k-1] + b[k]
            h_k = activation_function(a_k)            
            h[k] = h_k
        
        y_hat= h_k
        
        self.sgd = h_k
        
        return(y_hat)
        
        
    def backprop(self):
        """ using calculated cost to tweak weights and bias"""
        
        cost = self.sgd
        for k in range(1,self.num_layers+1):

            dEdwk = cost           
            self.w[k] = self.w[k] - self.lr * dEdwk

        pass
        
        
        
        
    def fit(X_train,y_train):
        
        
        pass
        
        
    


#def cross_entropy_loss(y,p,M):
#    loss = 0
#    observations = y.shape[0]
#    for i in range(M):
#        for o in range(observations):
#            loss-= y*np.log(p)
#            
#    
#    return(loss)

#TODO : functions to code or remove
def gradient():
    pass

def f():
    pass

def loss(y,y2):
    
    return(np.mean((y-y2)**2))
    
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
    J = loss(y,y_hat) #+ lamb * regularizer(W,b)
    
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

