# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 18:37:51 2020

@author: 49009427
"""


# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 17:39:11 2020

@author: 49009427
"""
import scipy as sp
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np





def infodistr(family):
    

    #=================================== 
    if family =='alpha':  ## [R+] ;     support [R+]
        
            p = 3
            typeofparams = 1 
            
            
    elif family =='argus':  ## [R+] ;     support ]0,1[
        
            p = 3
            typeofparams = 1 
            
            
    elif family =='beta':  ## [R+, R+] ;     support [0, 1]
        
            p = 4
            typeofparams = 2
            
            
    elif family =='betaprime':  ## [R+, R+] ;     support [R+]
        
            p = 4
            typeofparams = 2
    
            
    elif family =='binom':  ## [R+] ;     support [N+]
        
            p = 1
            typeofparams = 111 
            
    elif family =='nbinom':  ## [R+] ;     support [N+]
        
            p = 1
            typeofparams = 111 
    
            
    elif family =='bradford':  ## [R+] ;     support [0, 1]
        
            p = 3
            typeofparams = 1 
            
            
    elif family =='burr':  ## [R+, R+] ;     support [R+]
        
            p = 4
            typeofparams = 2
            
    
    elif family =='burr12':  ## [R+, R+] ;     support [R+]
        
            p = 4
            typeofparams = 2
            
            
    elif family =='chi':  ## [R+] ;     support [R+]
        
            p = 3
            typeofparams = 1 
            
            
    elif family =='chi2':  ## [R+] ;     support [R+]
        
            p = 3
            typeofparams = 1 
            
            
    elif family =='dgamma':  ## [R+] ;     support [R]
        
            p = 3
            typeofparams = 1 
            
            
    elif family =='dweibull':  ## [R+] ;     support [R]
        
            p = 3
            typeofparams = 1 
            
            
    elif family =='ev':  ##  []   support [R]
        
            p = 2
            typeofparams = 0
            
            
    elif family =='expon':  ## [] ;     support [R+]
        
            p = 2
            typeofparams = 0
            
            
    elif family =='exponnorm':  ## [R+] ;     support [R]
        
            p = 3
            typeofparams = 1 
            
            
    elif family =='exponweib':  ## [R+, R+] ;     support [R+]
        
            p = 4
            typeofparams = 2
            
        
    elif family =='exponpow':  ## [R+] ;     support [R+]
        
            p = 3
            typeofparams = 1 
            
            
    elif family =='f':  ## [R+, R+] ;     support [R+]
        
            p = 4
            typeofparams = 2
            
            
    elif family =='fatiguelife':  ## [R+] ;     support [R+]
        
            p = 3
            typeofparams = 1 
            
            
    elif family =='fisk':  ## [R+] ;     support [R+]
        
            p = 3
            typeofparams = 1 
            
            
    elif family =='gamma':  ##  [R+]   support [R+]
        
            p = 3
            typeofparams = 1 
            
            
    elif family =='genlogistic':  ## [R+]   support [R+]
        
            p = 3
            typeofparams = 1 
            
    
    elif family =='gennorm':  ##  [R+]   support [R]
        
            p = 3
            typeofparams = 1 
            
    elif family =='genexpon':  ##  [R+, R+, R+]   support [R+]
        
            p = 5
            typeofparams = 5
            
            
    elif family =='gengamma':  ##  [R+, R+]   support [R+]
        
            p = 4
            typeofparams = 2
            
            
    elif family =='geninvgauss':  ##  [R, R+]   support [R+]
        
            p = 4
            typeofparams = 3
            
        
    elif family =='geom':  ##  [R+]   support [N+]
        
            p = 1
            typeofparams = 111 
            
    
    elif family =='gompertz':  ##  [R+]   support [R+]
        
            p = 3
            typeofparams = 1 
            
            
    elif family =='gumbel_r':  ##  []   support [R+]
        
            p = 2
            typeofparams = 0
            
            
    elif family =='gumbel_l':  ##  []   support [R+]
        
            p = 2
            typeofparams = 0
            
            
    elif family =='invgamma':  ##  [R+]   support [R+]
        
            p = 3
            typeofparams = 1 
            
            
    elif family =='invgauss':  ##  [R+]   support [R+]
        
            p = 3
            typeofparams = 1 
            
            
    elif family =='invweibull':  ##  [R+]   support [R+]
        
            p = 3
            typeofparams = 1 
            
            
    elif family =='johnsonsu':  ##  [R+, R+]   support [R+]
        
            p = 4
            typeofparams = 2
            
            
    elif family =='laplace':  ##  []   support [R]
        
            p = 2
            typeofparams = 0
            
            
    elif family =='levy':  ##  []   support [R+]
        
            p = 2
            typeofparams = 0
            
            
    elif family =='levy_l':  ##  []   support [R-]
        
            p = 2
            typeofparams = 0
        
            
    elif family =='logistic':  ##  []   support [R]
        
            p = 2
            typeofparams = 0
            
           
    elif family =='loggamma':  ##  [R+]   support [R+]
        
            p = 3
            typeofparams = 1 
            
            
    elif family =='loglaplace':  ##  [R+]   support [R+]
        
            p = 3
            typeofparams = 1 
            
            
    elif family =='lognorm':  ##  [R+]   support [R+]
        
            p = 3
            typeofparams = 1 
            
            
    elif family =='lomax':  ##  [R+]   support [R+]
        
            p = 3
            typeofparams = 1 
            
            
    elif family =='maxwell':  ##  []   support [R+]
        
            p = 2
            typeofparams = 0
            
            
    elif family =='mielke':  ##  [R+, R+]   support [R+]
        
            p = 4
            typeofparams = 2
            
            
    elif family =='moyal':  ##  []   support [R]
        
            p = 2
            typeofparams = 0
            
    
    elif family =='nakagami':  ##  [R+]   support [R+]
        
            p = 3
            typeofparams = 1 
            
            
    elif family =='ncf':  ##  [R+, R+, R]   support [R+]
        
            p = 5
            typeofparams = 6
            
    elif family =='nct':  ##  [R+, R]   support [R+]
        
            p = 4
            typeofparams = 4
            
            
    elif family =='ncx2':  ##  [R+, R]   support [R+]
        
            p = 4
            typeofparams = 4
            
            
    elif family == 'norm':   ## [] ;     support [R]
        
            p = 2
            typeofparams = 0
            
     
    elif family == 'norminvgauss':   ## [R+, abs(b)>a] ;     support [R]
        
            p = 4
            typeofparams = 7
            
    
    elif family =='poisson':  ##  [R+]   support [N+]
        
            p = 1
            typeofparams = 111 
            
    
    elif family =='powerlaw':  ##  [R+]   support [0,1]
        
            p = 3
            typeofparams = 1 
            
        
    elif family =='powerlognorm':  ##  [R+, R+]   support [R+]
        
            p = 4
            typeofparams = 2
            
    
    elif family =='powernorm':  ##  [R+]   support [R+]
        
            p = 3
            typeofparams = 1 
            
            
    elif family =='rayleigh':  ##  []   support [R+]
        
            p = 2
            typeofparams = 0
            
            
    elif family =='rice':  ##  [R+]   support [R+]
        
            p = 3
            typeofparams = 1 
     
        
    elif family =='recipinvgauss':  ##  []   support [R+]
        
            p = 3
            typeofparams = 11
    
        
    elif family =='skewnorm':  ##  [R+]   support [R]
        
            p = 3
            typeofparams = 1 
            
            
    elif family =='t':  ##  [R+]   support [R]
        
            p = 3
            typeofparams = 1 
            
        
    elif family =='tukeylambda':  ##  [R]   support [R]
        
            p = 3
            typeofparams = 11
            
            
    
    elif family =='wald':  ##  []   support [R+]
        
            p = 2
            typeofparams = 0
            
               
        

    #=================================== 
    return(p, typeofparams)