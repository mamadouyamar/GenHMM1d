# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 18:24:59 2020

@author: 49009427
"""
import scipy as sp
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np

from SimMarkovChain import SimMarkovChain






def SimHMMGen(Q, family, theta, n, ntrial=0):
    
    r = Q.shape[0]
    MC = SimMarkovChain(Q,n,1)

    sim = np.zeros((n,r))
    simdata = np.zeros((n,1))
 
    #=================================== 
    if family =='alpha':  ## [R+] ;     support [R+]
        for j in range(r):
            sim[0:n,j] = stats.alpha.rvs(theta[j,0], loc=theta[j,1],
                                         scale=theta[j,2], size=n)  
            
            
    elif family =='argus':  ## [R+] ;     support ]0,1[
        for j in range(r):
            sim[0:n,j] = stats.argus.rvs(theta[j,0], loc=theta[j,1],
                                         scale=theta[j,2], size=n)     
            
            
    elif family =='beta':  ## [R+, R+] ;     support [0, 1]
        for j in range(r):
            sim[0:n,j] = stats.beta.rvs(theta[j,0], theta[j,1],
                                        loc=theta[j,2], scale=theta[j,3], size=n)  
            
            
    elif family =='betaprime':  ## [R+, R+] ;     support [R+]
        for j in range(r):
            sim[0:n,j] = stats.betaprime.rvs(theta[j,0], theta[j,1],
                                             loc=theta[j,2], scale=theta[j,3], size=n)  
            
    
    elif family =='binom':  ## [R+] ;     support [N+]
        for j in range(r):
            sim[0:n,j] = stats.binom.rvs(ntrial, theta[j,0], size=n)  
            
            
    elif family =='nbinom':  ## [R+] ;     support [N+]
        for j in range(r):
            sim[0:n,j] = stats.nbinom.rvs(ntrial, theta[j,0], size=n)  
            
            
    elif family =='bradford':  ## [R+] ;     support [0, 1]
        for j in range(r):
            sim[0:n,j] = stats.bradford.rvs(theta[j,0], loc=theta[j,1],
                                            scale=theta[j,2], size=n)   
            
            
    elif family =='burr':  ## [R+, R+] ;     support [R+]
        for j in range(r):
            sim[0:n,j] = stats.burr.rvs(theta[j,0], theta[j,1],
                                        loc=theta[j,2], scale=theta[j,3], size=n) 
            
    
    elif family =='burr12':  ## [R+, R+] ;     support [R+]
        for j in range(r):
            sim[0:n,j] = stats.burr12.rvs(theta[j,0], theta[j,1],
                                          loc=theta[j,2], scale=theta[j,3], size=n) 
            
            
    elif family =='chi':  ## [R+] ;     support [R+]
        for j in range(r):
            sim[0:n,j] = stats.chi.rvs(theta[j,0], loc=theta[j,1],
                                       scale=theta[j,2], size=n) 
            
            
    elif family =='chi2':  ## [R+] ;     support [R+]
        for j in range(r):
            sim[0:n,j] = stats.chi2.rvs(theta[j,0], loc=theta[j,1],
                                        scale=theta[j,2], size=n) 
             
            
            
    elif family =='dgamma':  ## [R+] ;     support [R]
        for j in range(r):
            sim[0:n,j] = stats.dgamma.rvs(theta[j,0], loc=theta[j,1],
                                          scale=theta[j,2], size=n)  
            
            
    elif family =='dweibull':  ## [R+] ;     support [R]
        for j in range(r):
            sim[0:n,j] = stats.dweibull.rvs(theta[j,0], loc=theta[j,1],
                                            scale=theta[j,2], size=n)  
            
            
    elif family =='ev':  ##  []   support [R]
        for j in range(r):
            sim[0:n,j] = stats.genextreme.rvs(loc=theta[j,0], scale=theta[j,1], size=n) 
            
            
    elif family =='expon':  ## [] ;     support [R+]
        for j in range(r):
            sim[0:n,j] = stats.expon.rvs(loc=theta[j,0], scale=theta[j,1],
                                         size=n)   
            
            
    elif family =='exponnorm':  ## [R+] ;     support [R]
        for j in range(r):
            sim[0:n,j] = stats.exponnorm.rvs(theta[j,0], loc=theta[j,1],
                                             scale=theta[j,2], size=n)    
            
            
    elif family =='exponweib':  ## [R+, R+] ;     support [R+]
        for j in range(r):
            sim[0:n,j] = stats.exponweib.rvs(theta[j,0], theta[j,1], loc=theta[j,2],
                                             scale=theta[j,3], size=n)    
            
        
    elif family =='exponpow':  ## [R+] ;     support [R+]
        for j in range(r):
            sim[0:n,j] = stats.exponpow.rvs(theta[j,0], loc=theta[j,1],
                                            scale=theta[j,2], size=n)   
            
            
    elif family =='f':  ## [R+, R+] ;     support [R+]
        for j in range(r):
            sim[0:n,j] = stats.f.rvs(theta[j,0], theta[j,1], loc=theta[j,2],
                                     scale=theta[j,3], size=n)   
            
            
    elif family =='fatiguelife':  ## [R+] ;     support [R+]
        for j in range(r):
            sim[0:n,j] = stats.fatiguelife.rvs(theta[j,0], loc=theta[j,1],
                                               scale=theta[j,2], size=n)  
            
            
    elif family =='fisk':  ## [R+] ;     support [R+]
        for j in range(r):
            sim[0:n,j] = stats.fisk.rvs(theta[j,0], loc=theta[j,1],
                                        scale=theta[j,2], size=n) 
            
            
    elif family =='gamma':  ##  [R+]   support [R+]
        for j in range(r):
            sim[0:n,j] = stats.gamma.rvs(theta[j,0], loc=theta[j,1],
                                         scale=theta[j,2], size=n) 
            
            
    elif family =='genlogistic':  ## [R+]   support [R+]
        for j in range(r):
            sim[0:n,j] = stats.genlogistic.rvs(theta[j,0], loc=theta[j,1],
                                               scale=theta[j,2], size=n)  
            
    
    elif family =='gennorm':  ##  [R+]   support [R]
        for j in range(r):
            sim[0:n,j] = stats.gennorm.rvs(theta[j,0], loc=theta[j,1],
                                           scale=theta[j,2], size=n) 
            
    elif family =='genexpon':  ##  [R+, R+, R+]   support [R+]
        for j in range(r):
            sim[0:n,j] = stats.genexpon.rvs(theta[j,0], theta[j,1], theta[j,2],
                                            loc=theta[j,3], scale=theta[j,4], size=n) 
            
            
    elif family =='gengamma':  ##  [R+, R+]   support [R+]
        for j in range(r):
            sim[0:n,j] = stats.gengamma.rvs(theta[j,0], theta[j,1],
                                            loc=theta[j,2], scale=theta[j,2], size=n) 
            
            
    elif family =='geninvgauss':  ##  [R, R+]   support [R+]
        for j in range(r):
            sim[0:n,j] = stats.geninvgauss.rvs(theta[j,0], theta[j,1],
                                               loc=theta[j,2], scale=theta[j,2], size=n) 
            
        
    elif family =='geom':  ##  [R+]   support [N+]
        for j in range(r):
            sim[0:n,j] = stats.geom.rvs(theta[j,0], size=n)  
            
    
    elif family =='gompertz':  ##  [R+]   support [R+]
        for j in range(r):
            sim[0:n,j] = stats.gompertz.rvs(theta[j,0], loc=theta[j,1],
                                            scale=theta[j,2], size=n) 
            
            
    elif family =='gumbel_r':  ##  []   support [R+]
        for j in range(r):
            sim[0:n,j] = stats.gumbel_r.rvs(loc=theta[j,0], scale=theta[j,1], size=n)   
            
            
    elif family =='gumbel_l':  ##  []   support [R+]
        for j in range(r):
            sim[0:n,j] = stats.gumbel_l.rvs(loc=theta[j,0], scale=theta[j,1], size=n) 
            
            
    elif family =='invgamma':  ##  [R+]   support [R+]
        for j in range(r):
            sim[0:n,j] = stats.invgamma.rvs(theta[j,0], loc=theta[j,1],
                                            scale=theta[j,2], size=n) 
            
            
    elif family =='invgauss':  ##  [R+]   support [R+]
        for j in range(r):
            sim[0:n,j] = stats.invgauss.rvs(theta[j,0], loc=theta[j,1],
                                            scale=theta[j,2], size=n) 
            
            
    elif family =='invweibull':  ##  [R+]   support [R+]
        for j in range(r):
            sim[0:n,j] = stats.invweibull.rvs(theta[j,0], loc=theta[j,1],
                                              scale=theta[j,2], size=n) 
            
            
    elif family =='johnsonsu':  ##  [R+, R+]   support [R+]
        for j in range(r):
            sim[0:n,j] = stats.johnsonsu.rvs(theta[j,0], theta[j,1],
                                             loc=theta[j,2], scale=theta[j,3], size=n) 
            
            
    elif family =='laplace':  ##  []   support [R]
        for j in range(r):
            sim[0:n,j] = stats.laplace.rvs(loc=theta[j,0], scale=theta[j,1], size=n) 
            
            
    elif family =='levy':  ##  []   support [R+]
        for j in range(r):
            sim[0:n,j] = stats.levy.rvs(loc=theta[j,0], scale=theta[j,1], size=n) 
            
            
    elif family =='levy_l':  ##  []   support [R-]
        for j in range(r):
            sim[0:n,j] = stats.levy_l.rvs(loc=theta[j,0], scale=theta[j,1], size=n) 
        
            
    elif family =='logistic':  ##  []   support [R]
        for j in range(r):
            sim[0:n,j] = stats.logistic.rvs(loc=theta[j,0], scale=theta[j,1], size=n) 
            
           
    elif family =='loggamma':  ##  [R+]   support [R+]
        for j in range(r):
            sim[0:n,j] = stats.loggamma.rvs(theta[j,0], loc=theta[j,1],
                                            scale=theta[j,2], size=n) 
            
            
    elif family =='loglaplace':  ##  [R+]   support [R+]
        for j in range(r):
            sim[0:n,j] = stats.loglaplace.rvs(theta[j,0], loc=theta[j,1],
                                              scale=theta[j,2], size=n) 
            
            
    elif family =='lognorm':  ##  [R+]   support [R+]
        for j in range(r):
            sim[0:n,j] = stats.loglaplace.rvs(theta[j,0], loc=theta[j,1],
                                              scale=theta[j,2], size=n) 
            
            
    elif family =='lomax':  ##  [R+]   support [R+]
        for j in range(r):
            sim[0:n,j] = stats.lomax.rvs(theta[j,0], loc=theta[j,1],
                                         scale=theta[j,2], size=n)    
            
            
    elif family =='maxwell':  ##  []   support [R+]
        for j in range(r):
            sim[0:n,j] = stats.maxwell.rvs(loc=theta[j,0], scale=theta[j,1], size=n) 
            
            
    elif family =='mielke':  ##  [R+, R+]   support [R+]
        for j in range(r):
            sim[0:n,j] = stats.mielke.rvs(theta[j,0], theta[j,1],
                                          loc=theta[j,2], scale=theta[j,3], size=n) 
            
            
    elif family =='moyal':  ##  []   support [R]
        for j in range(r):
            sim[0:n,j] = stats.moyal.rvs(loc=theta[j,0], scale=theta[j,1], size=n) 
            
    
    elif family =='nakagami':  ##  [R+]   support [R+]
        for j in range(r):
            sim[0:n,j] = stats.nakagami.rvs(theta[j,0], loc=theta[j,1],
                                            scale=theta[j,2], size=n)    
            
            
    elif family =='ncf':  ##  [R+, R+, R]   support [R+]
        for j in range(r):
            sim[0:n,j] = stats.ncf.rvs(theta[j,0], theta[j,1], theta[j,2],
                                       loc=theta[j,3], scale=theta[j,4], size=n) 
            
            
    elif family =='nct':  ##  [R+, R]   support [R+]
        for j in range(r):
            sim[0:n,j] = stats.nct.rvs(theta[j,0], theta[j,1],
                                       loc=theta[j,2], scale=theta[j,3], size=n) 
            
            
    elif family =='ncx2':  ##  [R+, R]   support [R+]
        for j in range(r):
            sim[0:n,j] = stats.ncx2.rvs(theta[j,0], theta[j,1],
                                        loc=theta[j,2], scale=theta[j,3], size=n) 
            
            
    elif family == 'norm':   ## [] ;     support [R]
        for j in range(r):
            sim[0:n,j] = stats.norm.rvs(loc=theta[j,0], scale=theta[j,1], size=n)  
            
            
     
    elif family == 'norminvgauss':   ## [R+, abs(b)>a] ;     support [R]
        for j in range(r):
            sim[0:n,j] = stats.norminvgauss.rvs(theta[j,0], theta[j,1],
                                                loc=theta[j,2], scale=theta[j,3], size=n)  
            
   
    elif family =='poisson':  ##  [R+]   support [N+]
        for j in range(r):
            sim[0:n,j] = stats.poisson.rvs(theta[j,0], size=n)  
            
            
    elif family =='powerlaw':  ##  [R+]   support [0,1]
        for j in range(r):
            sim[0:n,j] = stats.powerlaw.rvs(theta[j,0], loc=theta[j,1],
                                            scale=theta[j,2], size=n)  
            
        
    elif family =='powerlognorm':  ##  [R+, R+]   support [R+]
        for j in range(r):
            sim[0:n,j] = stats.powerlognorm.rvs(theta[j,0], theta[j,1],
                                                loc=theta[j,2], scale=theta[j,3], size=n) 
            
    
    elif family =='powernorm':  ##  [R+]   support [R+]
        for j in range(r):
            sim[0:n,j] = stats.powernorm.rvs(theta[j,0], loc=theta[j,1],
                                             scale=theta[j,2], size=n) 
            
            
    elif family =='rayleigh':  ##  []   support [R+]
        for j in range(r):
            sim[0:n,j] = stats.rayleigh.rvs(loc=theta[j,0], scale=theta[j,1], size=n) 
            
            
    elif family =='rice':  ##  [R+]   support [R+]
        for j in range(r):
            sim[0:n,j] = stats.rice.rvs(theta[j,0], loc=theta[j,1],
                                        scale=theta[j,2], size=n) 
     
        
    elif family =='recipinvgauss':  ##  [R]   support [R+]
        for j in range(r):
            sim[0:n,j] = stats.recipinvgauss.rvs(theta[j,0], loc=theta[j,1], scale=theta[j,2], size=n) 
            
    
        
    elif family =='skewnorm':  ##  [R+]   support [R]
        for j in range(r):
            sim[0:n,j] = stats.skewnorm.rvs(theta[j,0], loc=theta[j,1],
                                            scale=theta[j,2], size=n) 
            
            
    elif family =='t':  ##  [R+]   support [R]
        for j in range(r):
            sim[0:n,j] = stats.t.rvs(theta[j][0], loc=theta[j][1],
                                     scale=theta[j][2], size=n) 
            
        
    elif family =='tukeylambda':  ##  [R]   support [R]
        for j in range(r):
            sim[0:n,j] = stats.tukeylambda.rvs(theta[j,0], loc=theta[j,1],
                                               scale=theta[j,2], size=n) 
            
            
    
    elif family =='wald':  ##  []   support [R+]
        for j in range(r):
            sim[0:n,j] = stats.wald.rvs(loc=theta[j,0], scale=theta[j,1], size=n)       
            
            
        
        
        
    #=================================== 
    for i in range(n):
        simdata[i] = sim[i,int(MC[i])]


    #=================================== 
    return(simdata, sim, MC)





