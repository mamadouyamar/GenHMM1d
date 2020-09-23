# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 22:06:11 2020

@author: 49009427
"""
# -*- coding: utf-8 -*-

import scipy as sp
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np




def PDF(family, y, param, ntrial=0):
    

    #=================================== 
    if family =='alpha':  ## [R+] ;     support [R+]
        
            f =  stats.alpha.pdf(y, param[0], loc=param[1],
                                         scale=param[2])  
            
            
    elif family =='argus':  ## [R+] ;     support ]0,1[
        
            f =  stats.argus.pdf(y, param[0], loc=param[1],
                                         scale=param[2])     
            
            
    elif family =='beta':  ## [R+, R+] ;     support [0, 1]
        
            f =  stats.beta.pdf(y, param[0], param[1],
                                        loc=param[2], scale=param[3])  
            
            
    elif family =='betaprime':  ## [R+, R+] ;     support [R+]
        
            f =  stats.betaprime.pdf(y, param[0], param[1],
                                             loc=param[2], scale=param[3])  
            
            
    elif family =='binom':  ## [R+] ;     support [N+]
        
            f =  stats.binom.pmf(y, ntrial, param[0]) 
            
            
    elif family =='nbinom':  ## [R+] ;     support [N+]
        
            f =  stats.nbinom.pmf(y, ntrial, param[0]) 
            
            
    elif family =='bradford':  ## [R+] ;     support [0, 1]
        
            f =  stats.bradford.pdf(y, param[0], loc=param[1],
                                            scale=param[2])   
            
            
    elif family =='burr':  ## [R+, R+] ;     support [R+]
        
            f =  stats.burr.pdf(y, param[0], param[1],
                                        loc=param[2], scale=param[3]) 
            
    
    elif family =='burr12':  ## [R+, R+] ;     support [R+]
        
            f =  stats.burr12.pdf(y, param[0], param[1],
                                          loc=param[2], scale=param[3]) 
            
            
    elif family =='chi':  ## [R+] ;     support [R+]
        
            f =  stats.chi.pdf(y, param[0], loc=param[1],
                                       scale=param[2]) 
            
            
    elif family =='chi2':  ## [R+] ;     support [R+]
        
            f =  stats.chi2.pdf(y, param[0], loc=param[1],
                                        scale=param[2]) 
            
 
            
            
    elif family =='dgamma':  ## [R+] ;     support [R]
        
            f =  stats.dgamma.pdf(y, param[0], loc=param[1],
                                          scale=param[2])  
            
            
    elif family =='dweibull':  ## [R+] ;     support [R]
        
            f =  stats.dweibull.pdf(y, param[0], loc=param[1],
                                            scale=param[2])  
            
            
    elif family =='ev':  ##  []   support [R]
        
            f =  stats.genextreme.pdf(y, loc=param[0], scale=param[1]) 
            
            
    elif family =='expon':  ## [R+] ;     support [R+]
        
            f =  stats.expon.pdf(y, loc=param[0], scale=param[1])   
            
            
    elif family =='exponnorm':  ## [R+] ;     support [R]
        
            f =  stats.exponnorm.pdf(y, param[0], loc=param[1],
                                             scale=param[2])    
            
            
    elif family =='exponweib':  ## [R+, R+] ;     support [R+]
        
            f =  stats.exponweib.pdf(y, param[0], param[1], loc=param[2],
                                             scale=param[3])    
            
        
    elif family =='exponpow':  ## [R+] ;     support [R+]
        
            f =  stats.exponpow.pdf(y, param[0], loc=param[1],
                                            scale=param[2])   
            
            
    elif family =='f':  ## [R+, R+] ;     support [R+]
        
            f =  stats.f.pdf(y, param[0], param[1], loc=param[2],
                                     scale=param[3])   
            
            
    elif family =='fatiguelife':  ## [R+] ;     support [R+]
        
            f =  stats.fatiguelife.pdf(y, param[0], loc=param[1],
                                               scale=param[2])  
            
            
    elif family =='fisk':  ## [R+] ;     support [R+]
        
            f =  stats.fisk.pdf(y, param[0], loc=param[1],
                                        scale=param[2]) 
            
            
    elif family =='gamma':  ##  [R+]   support [R+]
        
            f =  stats.gamma.pdf(y, param[0], loc=param[1],
                                         scale=param[2]) 
            
            
    elif family =='genlogistic':  ## [R+]   support [R+]
        
            f =  stats.genlogistic.pdf(y, param[0], loc=param[1],
                                               scale=param[2])  
            
    
    elif family =='gennorm':  ##  [R+]   support [R]
        
            f =  stats.gennorm.pdf(y, param[0], loc=param[1],
                                           scale=param[2]) 
            
    elif family =='genexpon':  ##  [R+, R+, R+]   support [R+]
        
            f =  stats.genexpon.pdf(y, param[0], param[1], param[2],
                                            loc=param[3], scale=param[4]) 
            
            
    elif family =='gengamma':  ##  [R+, R+]   support [R+]
        
            f =  stats.gengamma.pdf(y, param[0], param[1],
                                            loc=param[2], scale=param[2]) 
            
            
    elif family =='geninvgauss':  ##  [R, R+]   support [R+]
        
            f =  stats.geninvgauss.pdf(y, param[0], param[1],
                                               loc=param[2], scale=param[2]) 
            
    elif family =='geom':  ##  [R+]   support [N+]

            f = stats.geom.pmf(y, param[0])  
            
    
    elif family =='gompertz':  ##  [R+]   support [R+]
        
            f =  stats.gompertz.pdf(y, param[0], loc=param[1],
                                            scale=param[2]) 
            
            
    elif family =='gumbel_r':  ##  []   support [R+]
        
            f =  stats.gumbel_r.pdf(y, loc=param[0], scale=param[1])   
            
            
    elif family =='gumbel_l':  ##  []   support [R+]
        
            f =  stats.gumbel_l.pdf(y, loc=param[0], scale=param[1]) 
            
            
    elif family =='invgamma':  ##  [R+]   support [R+]
        
            f =  stats.invgamma.pdf(y, param[0], loc=param[1],
                                            scale=param[2]) 
            
            
    elif family =='invgauss':  ##  [R+]   support [R+]
        
            f =  stats.invgauss.pdf(y, param[0], loc=param[1],
                                            scale=param[2]) 
            
            
    elif family =='invweibull':  ##  [R+]   support [R+]
        
            f =  stats.invweibull.pdf(y, param[0], loc=param[1],
                                              scale=param[2]) 
            
            
    elif family =='johnsonsu':  ##  [R+, R+]   support [R+]
        
            f =  stats.johnsonsu.pdf(y, param[0], param[1],
                                             loc=param[2], scale=param[3]) 
            
            
    elif family =='laplace':  ##  []   support [R]
        
            f =  stats.laplace.pdf(y, loc=param[0], scale=param[1]) 
            
            
    elif family =='levy':  ##  []   support [R+]
        
            f =  stats.levy.pdf(y, loc=param[0], scale=param[1]) 
            
            
    elif family =='levy_l':  ##  []   support [R-]
        
            f =  stats.levy_l.pdf(y, loc=param[0], scale=param[1]) 
        
            
    elif family =='logistic':  ##  []   support [R]
        
            f =  stats.logistic.pdf(y, loc=param[0], scale=param[1]) 
            
           
    elif family =='loggamma':  ##  [R+]   support [R+]
        
            f =  stats.loggamma.pdf(y, param[0], loc=param[1],
                                            scale=param[2]) 
            
            
    elif family =='loglaplace':  ##  [R+]   support [R+]
        
            f =  stats.loglaplace.pdf(y, param[0], loc=param[1],
                                              scale=param[2]) 
            
            
    elif family =='lognorm':  ##  [R+]   support [R+]
        
            f =  stats.loglaplace.pdf(y, param[0], loc=param[1],
                                              scale=param[2]) 
            
            
    elif family =='lomax':  ##  [R+]   support [R+]
        
            f =  stats.lomax.pdf(y, param[0], loc=param[1],
                                         scale=param[2])    
            
            
    elif family =='maxwell':  ##  []   support [R+]
        
            f =  stats.maxwell.pdf(y, loc=param[0], scale=param[1]) 
            
            
    elif family =='mielke':  ##  [R+, R+]   support [R+]
        
            f =  stats.mielke.pdf(y, param[0], param[1],
                                          loc=param[2], scale=param[3]) 
            
            
    elif family =='moyal':  ##  []   support [R]
        
            f =  stats.moyal.pdf(y, loc=param[0], scale=param[1]) 
            
    
    elif family =='nakagami':  ##  [R+]   support [R+]
        
            f =  stats.nakagami.pdf(y, param[0], loc=param[1],
                                            scale=param[2])    
            
            
    elif family =='ncf':  ##  [R+, R+, R]   support [R+]
        
            f =  stats.ncf.pdf(y, param[0], param[1], param[2],
                                       loc=param[3], scale=param[4]) 
            
            
    elif family =='nct':  ##  [R+, R]   support [R+]
        
            f =  stats.nct.pdf(y, param[0], param[1],
                                       loc=param[2], scale=param[3]) 
            
            
    elif family =='ncx2':  ##  [R+, R]   support [R+]
        
            f =  stats.ncx2.pdf(y, param[0], param[1],
                                        loc=param[2], scale=param[3]) 
            
            
    elif family == 'norm':   ## [] ;     support [R]
        
            f =  stats.norm.pdf(y, loc=param[0], scale=param[1])  
            
            
     
    elif family == 'norminvgauss':   ## [R+, abs(b)>a] ;     support [R]
        
            f =  stats.norminvgauss.pdf(y, param[0], param[1],
                                                loc=param[2], scale=param[3])  
            
            
    elif family =='poisson':  ##  [R+]   support [N+]

            f = stats.poisson.pmf(y, param[0])  
            
            
    elif family =='powerlaw':  ##  [R+]   support [0,1]
        
            f =  stats.powerlaw.pdf(y, param[0], loc=param[1],
                                            scale=param[2])  
            
        
    elif family =='powerlognorm':  ##  [R+, R+]   support [R+]
        
            f =  stats.powerlognorm.pdf(y, param[0], param[1],
                                                loc=param[2], scale=param[3]) 
            
    
    elif family =='powernorm':  ##  [R+]   support [R+]
        
            f =  stats.powernorm.pdf(y, param[0], loc=param[1],
                                             scale=param[2]) 
            
            
    elif family =='rayleigh':  ##  []   support [R+]
        
            f =  stats.rayleigh.pdf(y, loc=param[0], scale=param[1]) 
            
            
    elif family =='rice':  ##  [R+]   support [R+]
        
            f =  stats.rice.pdf(y, param[0], loc=param[1],
                                        scale=param[2]) 
     
        
    elif family =='recipinvgauss':  ##  []   support [R+]
        
            f =  stats.recipinvgauss.pdf(y, param[0], loc=param[1], scale=param[2]) 
            
    
        
    elif family =='skewnorm':  ##  [R+]   support [R]
        
            f =  stats.skewnorm.pdf(y, param[0], loc=param[1],
                                            scale=param[2]) 
            
            
    elif family =='t':  ##  [R+]   support [R]
        
            f =  stats.t.pdf(y, param[0], loc=param[1],
                                     scale=param[2]) 
            
        
    elif family =='tukeylambda':  ##  [R]   support [R]
        
            f =  stats.tukeylambda.pdf(y, param[0], loc=param[1],
                                               scale=param[2]) 
            
            
    
    elif family =='wald':  ##  []   support [R+]
        
            f =  stats.wald.pdf(y, loc=param[0], scale=param[1])       
            
               
        

    #=================================== 
    return(f)