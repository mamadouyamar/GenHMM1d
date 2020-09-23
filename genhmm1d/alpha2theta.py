# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 22:15:31 2020

@author: 49009427
"""
import scipy as sp
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np



def alpha2theta(family,param,typeofparams):
    
    ## the typeofparams are :
    
    #  [] == 0
    #  [R+] == 1 
    #  [R+] and discrete == 111 
    #  [R] == 11
    #  [R+, R+] == 2
    #  [R, R+] == 3
    #  [R+, R] == 4
    #  [R+, R+, R+] == 5
    #  [R+, R+, R] == 6
    
    #  [R+, R with abs(b)>a] == 7
    
    
    if typeofparams == 0:       ##  []
        theta = np.zeros((1,2))
        theta[0,0] = param[0]
        theta[0,1] = np.exp(param[1])
        
        
    elif typeofparams == 1:     ##  [R+]
        theta = np.zeros((1,3))
        theta[0,0] = np.exp(param[0])
        theta[0,1] = param[1]
        theta[0,2] = np.exp(param[2])
        
    elif typeofparams == 111:     ##  [R+]
        theta = np.zeros((1,1))
        theta[0,0] = np.exp(param[0])
    
        
    elif typeofparams == 11:     ##  [R]
        theta = np.zeros((1,3))
        theta[0,0] = param[0]
        theta[0,1] = param[1]
        theta[0,2] = np.exp(param[2])
        
        
    elif typeofparams == 2:          ##  [R+, R+]
        theta = np.zeros((1,4))
        theta[0,0:2] = np.exp(param[0:2])
        theta[0,2] = param[2]
        theta[0,3] = np.exp(param[3])
        
        
    elif typeofparams == 3:              ##  [R, R+]
        theta = np.zeros((1,4))
        theta[0,0] = param[0]
        theta[0,1] = np.exp(param[1])
        theta[0,2] = param[2]
        theta[0,3] = np.exp(param[3])
        
        
    elif typeofparams == 4:               ##  [R+, R]             
        theta = np.zeros((1,4))
        theta[0,0] = np.exp(param[0])
        theta[0,1] = param[1]
        theta[0,2] = param[2]
        theta[0,3] = np.exp(param[3])
        
        
    elif typeofparams == 5:              ##  [R+, R+, R+]
        theta = np.zeros((1,5))
        theta[0,0:3] = np.exp(param[0:3])
        theta[0,3] = param[3]
        theta[0,4] = np.exp(param[4])
        
        
    elif typeofparams == 6:              ##  [R+, R+, R]
        theta = np.zeros((1,5))
        theta[0,0:2] = np.exp(param[0:2])
        theta[0,2] = param[2]
        theta[0,3] = param[3]
        theta[0,4] = np.exp(param[4])
        
    
    elif typeofparams == 7:              ##  [R+, R with abs(b)>a]
        theta = np.zeros((1,4))
        theta[0,0] = np.exp(param[0])
        theta[0,1] = theta[0] * ( np.exp(2*param[1])-1 ) / ( np.exp(2*param[1])+1 )
        theta[0,2] = param[2]
        theta[0,3] = np.exp(param[3])
        
        



    return(theta)
