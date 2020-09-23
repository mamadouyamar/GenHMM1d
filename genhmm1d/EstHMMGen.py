# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 18:18:22 2020

@author: 49009427
"""
import scipy as sp
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.optimize import minimize
import pandas as pd
from joblib import Parallel, delayed
import multiprocessing
from scipy.optimize import basinhopping


from fitdistr import fitdistr
from infodistr import infodistr
from theta2alpha import theta2alpha
from alpha2theta import alpha2theta
from PDFunc import PDFunc
from CDF import CDF




def LLEMStep(y,family,lambda_EM,theta):
    
    LL = -sum( np.multiply(lambda_EM, np.squeeze(np.log(PDFunc(family, y, theta)), -1) ) )
    
    return(LL)



##=============================================================================

def EMStep(y, family, theta, Q, ntrial):
    
    n = len(y)
    r, p = theta.shape
    eta_bar_EM = np.zeros((n,r))
    eta_EM = np.zeros((n,r))
    lambda_EM = np.zeros((n,r))
    Lambda_EM = np.zeros((n,r))
    f = np.zeros((n,r))
    Z = np.zeros((n,1))
    Lambda_EM = np.zeros((r,r,n))


    for j in range(r):
        f[0:n,j] = PDFunc(family, y, theta[j,0:p], ntrial).reshape(n)
    #np.where(f[0:n,1] == 0)
    #np.where(f[0:n,0] == )
    #y[621]
    
    ## eta_bar_EM
    #dd = []
    eta_bar_EM[n-1,0:r] = 1/r
    for k in range(n-1):
        i = n-2-k
        j = i+1
        v = np.multiply(eta_bar_EM[j,0:r], f[j,0:r]).dot(np.transpose(Q))
        #dd.append(sum(v))
        eta_bar_EM[i,0:r] = v/sum(v)


    
    ## eta_EM
    eta0 = np.ones((1,r))/r
    v = np.multiply( ( eta0.dot(Q) ), f[0,0:r])
    Z[0] = sum(sum(v))
    eta_EM[0,0:r] = v/Z[0]

    for i in range(1,n):
        v = np.multiply( ( eta_EM[i-1,0:r].dot(Q) ), f[i,0:r] )
        Z[i] = sum(v)
        eta_EM[i,0:r] = v/Z[i]
    
    LL = sum(np.log(Z))
  

      
    ## lambda_EM
    v = np.multiply(eta_EM, eta_bar_EM)
    sv0 = np.sum(v, axis=-1)
    
    for j in range(r):
        lambda_EM[0:n,j] = np.divide(v[0:n,j] , sv0)

        
    ## Lambda 
    gc = np.multiply(eta_bar_EM, f)
    
    M = np.multiply(Q, np.multiply(np.transpose(eta0), gc[0,0:r]) )
    MM = sum(sum(M))
    Lambda_EM[0:r,0:r,0] = M/MM
        
    for i in range(1,n):
        eta_reshape = np.transpose(np.expand_dims(eta_EM[i-1,0:r], 0))
        gc_reshape = np.expand_dims(gc[i,0:r], 0)
        M = np.multiply( Q , eta_reshape.dot(gc_reshape) )
        MM = sum(sum(M))
        Lambda_EM[0:r,0:r,i] = M/MM

    nu_EM = np.mean(lambda_EM)

    Qnew_EM = Q
    
    for j in range(r):
        sv = np.sum(Lambda_EM[j,0:r,0:n], axis=-1)
        ssv = sum(sv)
        Qnew_EM[j,0:r] = sv/ssv
           
    theta_new_EM = theta
    for i in range(r):
        ## CA PREND TROP DE TEMPS SANS DOUTE A CAUSE DE LA FONCTION LAMBDA ??
        
        fun = lambda thetaa : -sum( np.multiply(lambda_EM[0:n,i],
                                                    np.squeeze(np.log(PDFunc(family, y, thetaa, ntrial)), -1) ) )
        res = minimize(fun, theta[i,0:p], method='Nelder-Mead')  # 'Nelder-Mead'
        theta_new_EM[i,0:p] = res.x    
        #optfunc = partial(LLEMStep, y,family,lambda_EM[0:n,i])
        #res = minimize(optfunc, theta[i,0:r], method='Nelder-Mead')
        #theta_new_EM[i,0:r] = res.x    
    #-sum(np.multiply(lambda_EM[0:n,1], np.squeeze(np.log(PDFunc(family, y, theta[1,0:p])), -1) ))
    #res = sp.optimize.minimize_scalar(fun, method='brent')
    
    #minimizer_kwargs = {"method": "BFGS"}
    #ret = basinhopping(fun, theta[0,0:p], minimizer_kwargs=minimizer_kwargs, niter=200)
    
    return (nu_EM, theta_new_EM, Qnew_EM, eta_EM, eta_bar_EM, lambda_EM, Lambda_EM, LL)
    


##=============================================================================

def Sn1d(U):
    
    n = len(U)
    u = np.sort(U)
    t = (-0.5 + np.arange(1,n+1) ) / n
    
    stat = (1/(12*n)) + sum( (u-t)**2 )
    
    return(stat)




##=============================================================================

def EstHMMGen(y, reg, family, max_iter, eps, ntrial=0):
    
    ninit = 100
    n = len(y)
    
    #n0 = math.floor(n/reg)
    #ind0 = np.arange(n0)    
    
    discreteFam = ['poisson', 'binom', 'geom', 'nbinom']
    
    
    p, typeofparams = infodistr(family)
    theta0 = np.zeros((reg,p))
    alpha0 = np.zeros((reg,p))
    
    for j in range(reg):
        #ind = j*n0+ind0
        #x = y[ind]
        x = y[np.random.choice(n, int(np.round(0.75*n)), replace=False)]
        if family not in discreteFam:
            x = np.append(x,min(y))
            x = np.append(x,max(y))
        tempFit = fitdistr(family, x, ntrial)
        theta0[j,0:p] = tempFit
        alpha0[j,0:p] = theta2alpha(family, tempFit, typeofparams)
    
        
    Q0 = np.ones((reg, reg))/reg


    for k in range(ninit):
        nu_EM, alpha_new_EM, Qnew_EM, eta_EM, eta_bar_EM, lambda_EM, Lambda_EM, LL = EMStep(y, family, alpha0, Q0, ntrial)
        Q0 = Qnew_EM
        alpha0 = alpha_new_EM
        #print(alpha0)
        #print(Q0)
        #print(k)
        
        
    for k in range(max_iter):
        nu_EM, alpha_new_EM, Qnew_EM, eta_EM, eta_bar_EM, lambda_EM, Lambda_EM, LL = EMStep(
            y, family, alpha0, Q0, ntrial)
        
        sum1 = sum(sum(abs(alpha0))) 
        sum2 = sum(sum(abs(alpha_new_EM-alpha0)))
        
        if (sum2 < sum1 * reg * eps):
            break
        
        Q0 = Qnew_EM
        alpha0 = alpha_new_EM
    
    
    alpha = alpha_new_EM
    Q = Qnew_EM
    
    theta = np.zeros((reg,p))
    for j in range(reg):
        theta[j,0:p] = alpha2theta(family,alpha[j,0:p],typeofparams)
    
    numel = theta.shape[0]*theta.shape[1]
    numParam = (numel+reg**2)
    
    AIC = (2 *  numParam - 2*LL) / n
    BIC = (np.log(n) *  numParam - 2*LL) / n
    CAIC = ( (np.log(n)+1) * numParam - 2*LL) / n
    AICc = AIC + (2*numParam*(numParam+1))/(n-numParam-1)
    HQC = (2*numParam*np.log(np.log(n)) - 2*LL)/n 
    
    
    cdf_gof = np.zeros((n, reg))

    

    if family in discreteFam:
        u_Ros = np.random.uniform(0,1,n)
        for j in range(reg):
            cdf_gof[0:n,j] = np.multiply((1-u_Ros), np.squeeze(CDF(family, y-0.7, theta[j,0:p],ntrial),-1)
                                         ) +  np.multiply(u_Ros, np.squeeze(CDF(family, y, theta[j,0:p],ntrial),-1))                                         
    else:
        for j in range(reg):
            cdf_gof[0:n,j] = np.squeeze(CDF(family, y, theta[j,0:p]),-1)
            
        
    eta00 = np.ones((1,reg))/reg
    w00 = np.concatenate((eta00, eta_EM), axis=0)
    W = w00[0:n,0:reg].dot(Q)
    U = np.sum( np.multiply(W, cdf_gof), -1 )
    
    cvm = Sn1d(U)
    
    
    return(theta, Q, eta_EM, nu_EM, U, cvm, W, lambda_EM, LL, AIC, BIC, CAIC, AICc, HQC)
    
    
    
    
    
    