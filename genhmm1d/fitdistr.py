# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 17:39:11 2020

@author: 49009427
"""
import scipy as sp
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np





def fitdistr(family, y, ntrial=0):
    

    #=================================== 
    if family =='alpha':  ## [R+] ;     support [R+]
        
            p1, loc, scale = stats.alpha.fit(y) 
            estimated_param = np.array([p1, loc, scale])
            
            
    elif family =='argus':  ## [R+] ;     support ]0,1[
        
            p1, loc, scale =  stats.argus.fit(y)  
            estimated_param = np.array([p1, loc, scale])
            
            
    elif family =='beta':  ## [R+, R+] ;     support [0, 1]
        
            p1, p2, loc, scale =  stats.beta.fit(y)  
            estimated_param = np.array([p1, p2, loc, scale])
            
            
    elif family =='betaprime':  ## [R+, R+] ;     support [R+]
        
            p1, p2, loc, scale = stats.betaprime.fit(y)  
            estimated_param = np.array([p1, p2, loc, scale])
            
    
    elif family =='binom':  ## [R+] ;     support [N+]

            p1 = np.mean(y/ntrial)  
            estimated_param = np.array([p1])
            
    elif family =='nbinom':  ## [R+] ;     support [N+]

            p1 = np.mean(y/ntrial)  
            estimated_param = np.array([p1])
            
    
    elif family =='bradford':  ## [R+] ;     support [0, 1]
        
            p1, loc, scale =  stats.bradford.fit(y)   
            estimated_param = np.array([p1, loc, scale])
            
            
    elif family =='burr':  ## [R+, R+] ;     support [R+]
        
            p1, p2, loc, scale = stats.burr.fit(y) 
            estimated_param = np.array([p1, p2, loc, scale])
            
    
    elif family =='burr12':  ## [R+, R+] ;     support [R+]
        
            p1, p2, loc, scale = stats.burr12.fit(y) 
            estimated_param = np.array([p1, p2, loc, scale])
            
            
    elif family =='chi':  ## [R+] ;     support [R+]
        
            p1, loc, scale = stats.chi.fit(y)
            estimated_param = np.array([p1, loc, scale])
            
            
    elif family =='chi2':  ## [R+] ;     support [R+]
        
            p1, loc, scale =  stats.chi2.fit(y) 
            estimated_param = np.array([p1, loc, scale])
            
            
    elif family =='dgamma':  ## [R+] ;     support [R]
        
            p1, loc, scale =  stats.dgamma.fit(y)  
            estimated_param = np.array([p1, loc, scale])
            
            
    elif family =='dweibull':  ## [R+] ;     support [R]
        
            p1, loc, scale =  stats.dweibull.fit(y)  
            estimated_param = np.array([p1, loc, scale])
            
            
    elif family =='ev':  ##  []   support [R]
        
            loc, scale = stats.genextreme.fit(y) 
            estimated_param = np.array([loc, scale])
            
            
    elif family =='expon':  ## [] ;     support [R+]
        
            loc, scale =  stats.expon.fit(y)   
            estimated_param = np.array([loc, scale])
            
            
    elif family =='exponnorm':  ## [R+] ;     support [R]
        
            p1, loc, scale =  stats.exponnorm.fit(y)    
            estimated_param = np.array([p1, loc, scale])
            
            
    elif family =='exponweib':  ## [R+, R+] ;     support [R+]
        
            p1, p2, loc, scale = stats.exponweib.fit(y)    
            estimated_param = np.array([p1, p2, loc, scale])
            
        
    elif family =='exponpow':  ## [R+] ;     support [R+]
        
            p1, loc, scale =  stats.exponpow.fit(y)   
            estimated_param = np.array([p1, loc, scale])
            
            
    elif family =='f':  ## [R+, R+] ;     support [R+]
        
            p1, p2, loc, scale = stats.f.fit(y)   
            estimated_param = np.array([p1, p2, loc, scale])
            
            
    elif family =='fatiguelife':  ## [R+] ;     support [R+]
        
            p1, loc, scale =  stats.fatiguelife.fit(y)  
            estimated_param = np.array([p1, loc, scale])
            
            
    elif family =='fisk':  ## [R+] ;     support [R+]
        
            p1, loc, scale =  stats.fisk.fit(y) 
            estimated_param = np.array([p1, loc, scale])
            
            
    elif family =='gamma':  ##  [R+]   support [R+]
        
            p1, loc, scale =  stats.gamma.fit(y) 
            estimated_param = np.array([p1, loc, scale])
            
            
    elif family =='genlogistic':  ## [R+]   support [R+]
        
            p1, loc, scale =  stats.genlogistic.fit(y)  
            estimated_param = np.array([p1, loc, scale])
            
    
    elif family =='gennorm':  ##  [R+]   support [R]
        
            p1, loc, scale = stats.gennorm.fit(y) 
            estimated_param = np.array([p1, loc, scale])
            
    elif family =='genexpon':  ##  [R+, R+, R+]   support [R+]
        
            p1, p2, p3, loc, scale = stats.genexpon.fit(y) 
            estimated_param = np.array([p1, p2, p3, loc, scale])
            
            
    elif family =='gengamma':  ##  [R+, R+]   support [R+]
        
            p1, p2, loc, scale = stats.gengamma.fit(y) 
            estimated_param = np.array([p1, p2, loc, scale])
            
            
    elif family =='geninvgauss':  ##  [R, R+]   support [R+]
        
            p1, p2, loc, scale = stats.geninvgauss.fit(y) 
            estimated_param = np.array([p1, p2, loc, scale])
            
            
    elif family =='geom':  ##  [R+]   support [N+]
        
            p1 = len(y)/sum(y)  
            estimated_param = np.array([p1])
            
    
    elif family =='gompertz':  ##  [R+]   support [R+]
        
            p1, loc, scale = stats.gompertz.fit(y) 
            estimated_param = np.array([p1, loc, scale])
            
            
    elif family =='gumbel_r':  ##  []   support [R+]
        
            loc, scale = stats.gumbel_r.fit(y)   
            estimated_param = np.array([loc, scale])
            
            
    elif family =='gumbel_l':  ##  []   support [R+]
        
            loc, scale = stats.gumbel_l.fit(y) 
            estimated_param = np.array([loc, scale])
            
            
    elif family =='invgamma':  ##  [R+]   support [R+]
        
            p1, loc, scale = stats.invgamma.fit(y) 
            estimated_param = np.array([p1, loc, scale])
            
            
    elif family =='invgauss':  ##  [R+]   support [R+]
        
            p1, loc, scale =  stats.invgauss.fit(y) 
            estimated_param = np.array([p1, loc, scale])
            
            
    elif family =='invweibull':  ##  [R+]   support [R+]
        
            p1, loc, scale =  stats.invweibull.fit(y) 
            estimated_param = np.array([p1, loc, scale])
            
            
    elif family =='johnsonsu':  ##  [R+, R+]   support [R+]
        
            p1, p2, loc, scale = stats.johnsonsu.fit(y) 
            estimated_param = np.array([p1, p2, loc, scale])
            
            
    elif family =='laplace':  ##  []   support [R]
        
            loc, scale = stats.laplace.fit(y) 
            estimated_param = np.array([loc, scale])
            
            
    elif family =='levy':  ##  []   support [R+]
        
            loc, scale =  stats.levy.fit(y) 
            estimated_param = np.array([loc, scale])
            
            
    elif family =='levy_l':  ##  []   support [R-]
        
            loc, scale =  stats.levy_l.fit(y) 
            estimated_param = np.array([loc, scale])
        
            
    elif family =='logistic':  ##  []   support [R]
        
            loc, scale = stats.logistic.fit(y) 
            estimated_param = np.array([loc, scale])
            
           
    elif family =='loggamma':  ##  [R+]   support [R+]
        
            p1, loc, scale =  stats.loggamma.fit(y) 
            estimated_param = np.array([p1, loc, scale])
            
            
    elif family =='loglaplace':  ##  [R+]   support [R+]
        
            p1, loc, scale =  stats.loglaplace.fit(y) 
            estimated_param = np.array([p1, loc, scale])
            
            
    elif family =='lognorm':  ##  [R+]   support [R+]
        
            p1, loc, scale =  stats.loglaplace.fit(y) 
            estimated_param = np.array([p1, loc, scale])
            
            
    elif family =='lomax':  ##  [R+]   support [R+]
        
            p1, loc, scale =  stats.lomax.fit(y)   
            estimated_param = np.array([p1, loc, scale])
            
            
    elif family =='maxwell':  ##  []   support [R+]
        
            loc, scale = stats.maxwell.fit(y) 
            estimated_param = np.array([loc, scale])
            
            
    elif family =='mielke':  ##  [R+, R+]   support [R+]
        
            p1, p2, loc, scale = stats.mielke.fit(y) 
            estimated_param = np.array([p1, p2, loc, scale])
            
            
    elif family =='moyal':  ##  []   support [R]
        
            loc, scale = stats.moyal.fit(y) 
            estimated_param = np.array([loc, scale])
            
    
    elif family =='nakagami':  ##  [R+]   support [R+]
        
            p1, loc, scale =  stats.nakagami.fit(y)   
            estimated_param = np.array([p1, loc, scale])
            
            
    elif family =='ncf':  ##  [R+, R+, R]   support [R+]
        
            p1, p2, p3, loc, scale = stats.ncf.fit(y) 
            estimated_param = np.array([p1, p2, p3, loc, scale])
            
    elif family =='nct':  ##  [R+, R]   support [R+]
        
            p1, p2, loc, scale = stats.nct.fit(y) 
            estimated_param = np.array([p1, p2, loc, scale])
            
            
    elif family =='ncx2':  ##  [R+, R]   support [R+]
        
            p1, p2, loc, scale = stats.ncx2.fit(y) 
            estimated_param = np.array([p1, p2, loc, scale])
            
            
    elif family == 'norm':   ## [] ;     support [R]
        
            loc, scale =  stats.norm.fit(y)  
            estimated_param = np.array([loc, scale])
            
     
    elif family == 'norminvgauss':   ## [R+, abs(b)>a] ;     support [R]
        
            p1, p2, loc, scale = stats.norminvgauss.fit(y)  
            estimated_param = np.array([p1, p2, loc, scale])
            
    
    elif family =='poisson':  ##  [R+]   support [N+]
        
            p1 = np.mean(y)  
            estimated_param = np.array([p1])        
    
    
    elif family =='powerlaw':  ##  [R+]   support [0,1]
        
            p1, loc, scale = stats.powerlaw.fit(y)  
            estimated_param = np.array([p1, loc, scale])
            
        
    elif family =='powerlognorm':  ##  [R+, R+]   support [R+]
        
            p1, p2, loc, scale = stats.powerlognorm.fit(y) 
            estimated_param = np.array([p1, p2, loc, scale])
            
    
    elif family =='powernorm':  ##  [R+]   support [R+]
        
            p1, loc, scale = stats.powernorm.fit(y) 
            estimated_param = np.array([p1, loc, scale])
            
    elif family =='rayleigh':  ##  []   support [R+]
        
            loc, scale = stats.rayleigh.fit(y) 
            estimated_param = np.array([loc, scale])
            
            
    elif family =='rice':  ##  [R+]   support [R+]
        
            p1, loc, scale =  stats.rice.fit(y) 
            estimated_param = np.array([p1, loc, scale])
     
        
    elif family =='recipinvgauss':  ##  []   support [R+]
        
            p1, loc, scale =  stats.recipinvgauss.fit(y) 
            estimated_param = np.array([p1, loc, scale])
    
        
    elif family =='skewnorm':  ##  [R+]   support [R]
        
            p1, loc, scale =  stats.skewnorm.fit(y) 
            estimated_param = np.array([p1, loc, scale])
            
            
    elif family =='t':  ##  [R+]   support [R]
        
            p1, loc, scale =  stats.t.fit(y) 
            estimated_param = np.array([p1, loc, scale])
            
        
    elif family =='tukeylambda':  ##  [R]   support [R]
        
            p1, loc, scale  =  stats.tukeylambda.fit(y) 
            estimated_param = np.array([p1, loc, scale])
            
            
    
    elif family =='wald':  ##  []   support [R+]
        
            loc, scale =  stats.wald.fit(y)    
            estimated_param = np.array([loc, scale])
            
               
        

    #=================================== 
    return(estimated_param)