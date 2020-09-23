# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 22:35:10 2020

@author: 49009427
"""

import scipy as sp
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np




def theta2alpha(family,param,typeofparams):
    
    ## the typeofparams are :
    
    #  [] == 0
    #  [R+] == 1 
    #  [R+] and discrete = 111
    #  [R] == 11
    #  [R+, R+] == 2
    #  [R, R+] == 3
    #  [R+, R] == 4
    #  [R+, R+, R+] == 5
    #  [R+, R+, R] == 6
    
    #  [R+, R with abs(b)>a] == 7
    
    
    if typeofparams == 0:       ##  []
        alpha = np.zeros((1,2))
        alpha[0,0] = param[0]
        alpha[0,1] = np.log(param[1])
        
        
    elif typeofparams == 1:     ##  [R+]
        alpha = np.zeros((1,3))
        alpha[0,0] = np.log(param[0])
        alpha[0,1] = param[1]
        alpha[0,2] = np.log(param[2])
        
        
    elif typeofparams == 111:     ##  [R+]
        alpha = np.zeros((1,1))
        alpha[0,0] = np.log(param[0])
        
        
    elif typeofparams == 11:     ##  [R]
        alpha = np.zeros((1,3))
        alpha[0,0] = param[0]
        alpha[0,1] = param[1]
        alpha[0,2] = np.log(param[2])
        
        
    elif typeofparams == 2:          ##  [R+, R+]
        alpha = np.zeros((1,4))
        alpha[0,0:2] = np.log(param[0:2])
        alpha[0,2] = param[2]
        alpha[0,3] = np.log(param[3])
        
        
    elif typeofparams == 3:              ##  [R, R+]
        alpha = np.zeros((1,4))
        alpha[0,0] = param[0]
        alpha[0,1] = np.log(param[1])
        alpha[0,2] = param[2]
        alpha[0,3] = np.log(param[3])
        
        
    elif typeofparams == 4:               ##  [R+, R]             
        alpha = np.zeros((1,4))
        alpha[0,0] = np.log(param[0])
        alpha[0,1] = param[1]
        alpha[0,2] = param[2]
        alpha[0,3] = np.log(param[3])
        
        
    elif typeofparams == 5:              ##  [R+, R+, R+]
        alpha = np.zeros((1,5))
        alpha[0,0:3] = np.log(param[0:3])
        alpha[0,3] = param[3]
        alpha[0,4] = np.log(param[4])
        
        
    elif typeofparams == 6:              ##  [R+, R+, R]
        alpha = np.zeros((1,5))
        alpha[0,0:2] = np.log(param[0,2])
        alpha[0,2] = param[2]
        alpha[0,3] = param[3]
        alpha[0,4] = np.log(param[4])
        
        
    elif typeofparams == 7:              ##  [R+, R with abs(b)>a]
        alpha = np.zeros((1,4))
        alpha[0,0] = np.log(param[0])
        alpha[0,1] = np.log( (param[1]+param[0]) / (param[0]-param[1]) )
        alpha[0,2] = param[2]
        alpha[0,3] = np.log(param[3])



    return(alpha)
