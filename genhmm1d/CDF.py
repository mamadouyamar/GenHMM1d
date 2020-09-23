# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 22:05:18 2020

@author: 49009427
"""
import scipy as sp
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np




def CDF(family, y, param, ntrial=0):
    

    #=================================== 
    if family =='alpha':  ## [R+] ;     support [R+]
        
            F = stats.alpha.cdf(y, param[0], loc=param[1],
                                         scale=param[2])  
            
            
    elif family =='argus':  ## [R+] ;     support ]0,1[
        
            F = stats.argus.cdf(y, param[0], loc=param[1],
                                         scale=param[2])     
            
            
    elif family =='beta':  ## [R+, R+] ;     support [0, 1]
        
            F = stats.beta.cdf(y, param[0], param[1],
                                        loc=param[2], scale=param[3])  
            
            
    elif family =='betaprime':  ## [R+, R+] ;     support [R+]
        
            F = stats.betaprime.cdf(y, param[0], param[1],
                                             loc=param[2], scale=param[3])  
            
            
    elif family =='binom':  ## [R+] ;     support [N+]
        
            F =  stats.binom.cdf(y, ntrial, param[0]) 
            
            
    elif family =='nbinom':  ## [R+] ;     support [N+]
        
            F =  stats.nbinom.cdf(y, ntrial, param[0]) 
            
            
    elif family =='bradford':  ## [R+] ;     support [0, 1]
        
            F = stats.bradford.cdf(y, param[0], loc=param[1],
                                            scale=param[2])   
            
            
    elif family =='burr':  ## [R+, R+] ;     support [R+]
        
            F = stats.burr.cdf(y, param[0], param[1],
                                        loc=param[2], scale=param[3]) 
            
    
    elif family =='burr12':  ## [R+, R+] ;     support [R+]
        
            F = stats.burr12.cdf(y, param[0], param[1],
                                          loc=param[2], scale=param[3]) 
            
            
    elif family =='chi':  ## [R+] ;     support [R+]
        
            F = stats.chi.cdf(y, param[0], loc=param[1],
                                       scale=param[2]) 
            
            
    elif family =='chi2':  ## [R+] ;     support [R+]
        
            F = stats.chi2.cdf(y, param[0], loc=param[1],
                                        scale=param[2]) 
             
            
            
    elif family =='dgamma':  ## [R+] ;     support [R]
        
            F = stats.dgamma.cdf(y, param[0], loc=param[1],
                                          scale=param[2])  
            
            
    elif family =='dweibull':  ## [R+] ;     support [R]
        
            F = stats.dweibull.cdf(y, param[0], loc=param[1],
                                            scale=param[2])  
            
            
    elif family =='ev':  ##  []   support [R]
        
            F = stats.genextreme.cdf(y, loc=param[0], scale=param[1]) 
            
            
    elif family =='expon':  ## [R+] ;     support [R+]
        
            F = stats.expon.cdf(y, loc=param[0], scale=param[1])   
            
            
    elif family =='exponnorm':  ## [R+] ;     support [R]
        
            F = stats.exponnorm.cdf(y, param[0], loc=param[1],
                                             scale=param[2])    
            
            
    elif family =='exponweib':  ## [R+, R+] ;     support [R+]
        
            F = stats.exponweib.cdf(y, param[0], param[1], loc=param[2],
                                             scale=param[3])    
            
        
    elif family =='exponpow':  ## [R+] ;     support [R+]
        
            F = stats.exponpow.cdf(y, param[0], loc=param[1],
                                            scale=param[2])   
            
            
    elif family =='f':  ## [R+, R+] ;     support [R+]
        
            F = stats.f.cdf(y, param[0], param[1], loc=param[2],
                                     scale=param[3])   
            
            
    elif family =='fatiguelife':  ## [R+] ;     support [R+]
        
            F = stats.fatiguelife.cdf(y, param[0], loc=param[1],
                                               scale=param[2])  
            
            
    elif family =='fisk':  ## [R+] ;     support [R+]
        
            F = stats.fisk.cdf(y, param[0], loc=param[1],
                                        scale=param[2]) 
            
            
    elif family =='gamma':  ##  [R+]   support [R+]
        
            F = stats.gamma.cdf(y, param[0], loc=param[1],
                                         scale=param[2]) 
            
            
    elif family =='genlogistic':  ## [R+]   support [R+]
        
            F = stats.genlogistic.cdf(y, param[0], loc=param[1],
                                               scale=param[2])  
            
    
    elif family =='gennorm':  ##  [R+]   support [R]
        
            F = stats.gennorm.cdf(y, param[0], loc=param[1],
                                           scale=param[2]) 
            
    elif family =='genexpon':  ##  [R+, R+, R+]   support [R+]
        
            F = stats.genexpon.cdf(y, param[0], param[1], param[2],
                                            loc=param[3], scale=param[4]) 
            
            
    elif family =='gengamma':  ##  [R+, R+]   support [R+]
        
            F = stats.gengamma.cdf(y, param[0], param[1],
                                            loc=param[2], scale=param[2]) 
            
            
    elif family =='geninvgauss':  ##  [R, R+]   support [R+]
        
            F = stats.geninvgauss.cdf(y, param[0], param[1],
                                               loc=param[2], scale=param[2]) 
            
            
    elif family =='geom':  ##  [R+]   support [N+]

            F = stats.geom.cdf(y, param[0])  
            
    
    elif family =='gompertz':  ##  [R+]   support [R+]
        
            F = stats.gompertz.cdf(y, param[0], loc=param[1],
                                            scale=param[2]) 
            
            
    elif family =='gumbel_r':  ##  []   support [R+]
        
            F = stats.gumbel_r.cdf(y, loc=param[0], scale=param[1])   
            
            
    elif family =='gumbel_l':  ##  []   support [R+]
        
            F = stats.gumbel_l.cdf(y, loc=param[0], scale=param[1]) 
            
            
    elif family =='invgamma':  ##  [R+]   support [R+]
        
            F = stats.invgamma.cdf(y, param[0], loc=param[1],
                                            scale=param[2]) 
            
            
    elif family =='invgauss':  ##  [R+]   support [R+]
        
            F = stats.invgauss.cdf(y, param[0], loc=param[1],
                                            scale=param[2]) 
            
            
    elif family =='invweibull':  ##  [R+]   support [R+]
        
            F = stats.invweibull.cdf(y, param[0], loc=param[1],
                                              scale=param[2]) 
            
            
    elif family =='johnsonsu':  ##  [R+, R+]   support [R+]
        
            F = stats.johnsonsu.cdf(y, param[0], param[1],
                                             loc=param[2], scale=param[3]) 
            
            
    elif family =='laplace':  ##  []   support [R]
        
            F = stats.laplace.cdf(y, loc=param[0], scale=param[1]) 
            
            
    elif family =='levy':  ##  []   support [R+]
        
            F = stats.levy.cdf(y, loc=param[0], scale=param[1]) 
            
            
    elif family =='levy_l':  ##  []   support [R-]
        
            F = stats.levy_l.cdf(y, loc=param[0], scale=param[1]) 
        
            
    elif family =='logistic':  ##  []   support [R]
        
            F = stats.logistic.cdf(y, loc=param[0], scale=param[1]) 
            
           
    elif family =='loggamma':  ##  [R+]   support [R+]
        
            F = stats.loggamma.cdf(y, param[0], loc=param[1],
                                            scale=param[2]) 
            
            
    elif family =='loglaplace':  ##  [R+]   support [R+]
        
            F = stats.loglaplace.cdf(y, param[0], loc=param[1],
                                              scale=param[2]) 
            
            
    elif family =='lognorm':  ##  [R+]   support [R+]
        
            F = stats.loglaplace.cdf(y, param[0], loc=param[1],
                                              scale=param[2]) 
            
            
    elif family =='lomax':  ##  [R+]   support [R+]
        
            F = stats.lomax.cdf(y, param[0], loc=param[1],
                                         scale=param[2])    
            
            
    elif family =='maxwell':  ##  []   support [R+]
        
            F = stats.maxwell.cdf(y, loc=param[0], scale=param[1]) 
            
            
    elif family =='mielke':  ##  [R+, R+]   support [R+]
        
            F = stats.mielke.cdf(y, param[0], param[1],
                                          loc=param[2], scale=param[3]) 
            
            
    elif family =='moyal':  ##  []   support [R]
        
            F = stats.moyal.cdf(y, loc=param[0], scale=param[1]) 
            
    
    elif family =='nakagami':  ##  [R+]   support [R+]
        
            F = stats.nakagami.cdf(y, param[0], loc=param[1],
                                            scale=param[2])    
            
            
    elif family =='ncf':  ##  [R+, R+, R]   support [R+]
        
            F = stats.ncf.cdf(y, param[0], param[1], param[2],
                                       loc=param[3], scale=param[4]) 
            
            
    elif family =='nct':  ##  [R+, R]   support [R+]
        
            F = stats.nct.cdf(y, param[0], param[1],
                                       loc=param[2], scale=param[3]) 
            
            
    elif family =='ncx2':  ##  [R+, R]   support [R+]
        
            F = stats.ncx2.cdf(y, param[0], param[1],
                                        loc=param[2], scale=param[3]) 
            
            
    elif family == 'norm':   ## [] ;     support [R]
        
            F = stats.norm.cdf(y, loc=param[0], scale=param[1])  
            
            
     
    elif family == 'norminvgauss':   ## [R+, abs(b)>a] ;     support [R]
        
            F = stats.norminvgauss.cdf(y, param[0], param[1],
                                                loc=param[2], scale=param[3])  
            
    
    elif family =='poisson':  ##  [R+]   support [N+]

            F = stats.poisson.cdf(y, param[0])  
            
            
    elif family =='powerlaw':  ##  [R+]   support [0,1]
        
            F = stats.powerlaw.cdf(y, param[0], loc=param[1],
                                            scale=param[2])  
            
        
    elif family =='powerlognorm':  ##  [R+, R+]   support [R+]
        
            F = stats.powerlognorm.cdf(y, param[0], param[1],
                                                loc=param[2], scale=param[3]) 
            
    
    elif family =='powernorm':  ##  [R+]   support [R+]
        
            F = stats.powernorm.cdf(y, param[0], loc=param[1],
                                             scale=param[2]) 
            
            
    elif family =='rayleigh':  ##  []   support [R+]
        
            F = stats.rayleigh.cdf(y, loc=param[0], scale=param[1]) 
            
            
    elif family =='rice':  ##  [R+]   support [R+]
        
            F = stats.rice.cdf(y, param[0], loc=param[1],
                                        scale=param[2]) 
     
        
    elif family =='recipinvgauss':  ##  []   support [R+]
        
            F = stats.recipinvgauss.cdf(y, param[0], loc=param[1], scale=param[2]) 
            
    
        
    elif family =='skewnorm':  ##  [R+]   support [R]
        
            F = stats.skewnorm.cdf(y, param[0], loc=param[1],
                                            scale=param[2]) 
            
            
    elif family =='t':  ##  [R+]   support [R]
        
            F = stats.t.cdf(y, param[0], loc=param[1],
                                     scale=param[2]) 
            
        
    elif family =='tukeylambda':  ##  [R]   support [R]
        
            F = stats.tukeylambda.cdf(y, param[0], loc=param[1],
                                               scale=param[2]) 
            
            
    
    elif family =='wald':  ##  []   support [R+]
        
            F = stats.wald.cdf(y, loc=param[0], scale=param[1])       
            
               
        

    #=================================== 
    return(F)