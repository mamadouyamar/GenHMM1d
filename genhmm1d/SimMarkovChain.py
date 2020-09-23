# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 18:24:06 2020

@author: 49009427
"""


import scipy as sp
import numpy as np


def SimMarkovChain(Q,n,eta0):
    r,p = Q.shape

    x = np.zeros((n,1))
    x0 = np.zeros((n,r))

    ind = eta0

    if r>1:
        for k in range(r):
            x0[0:n,k] = np.random.choice(r, n, p=Q[k,])
        for i in range(n):
            x[i] = x0[i][int(ind)]
            ind = int(x[i])
    else:
        x[0:n] = 0   

    MC = x

    return(MC)







