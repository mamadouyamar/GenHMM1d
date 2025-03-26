# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 23:13:28 2020

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

class HMM:
    def alpha2theta(self, param, typeofparams):


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



    ##=============================================================================
    ##=============================================================================
    ##=============================================================================
    ##=============================================================================
    def CDF(self, family, y, param, ntrial=0):


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




    ##=============================================================================
    ##=============================================================================
    ##=============================================================================
    ##=============================================================================
    def fitdistr(self, family, y, ntrial=0):


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


        elif family =='skewnorm':  ##  [R]   support [R]

                p1, loc, scale = stats.skewnorm.fit(y)
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




    ##=============================================================================
    ##=============================================================================
    ##=============================================================================
    ##=============================================================================
    def infodistr(self, family):


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


        elif family =='skewnorm':  ##  [R]   support [R]

                p = 3
                typeofparams = 11


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



    ##=============================================================================
    ##=============================================================================
    ##=============================================================================
    ##=============================================================================
    def PDF(self, family, y, param, ntrial=0):


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



        elif family =='skewnorm':  ##  [R]   support [R]

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



    ##=============================================================================
    ##=============================================================================
    ##=============================================================================
    ##=============================================================================
    def PDFunc(self, family, y, param, ntrial=0):


        #===================================
        if family =='alpha':  ## [R+] ;     support [R+]

                f =  stats.alpha.pdf(y, np.exp(param[0]), loc=param[1],
                                             scale=np.exp(param[2]))


        elif family =='argus':  ## [R+] ;     support ]0,1[

                f =  stats.argus.pdf(y, np.exp(param[0]), loc=param[1],
                                             scale=np.exp(param[2]))


        elif family =='beta':  ## [R+, R+] ;     support [0, 1]

                f =  stats.beta.pdf(y, np.exp(param[0]), np.exp(param[1]),
                                            loc=param[2], scale=np.exp(param[3]))


        elif family =='betaprime':  ## [R+, R+] ;     support [R+]

                f =  stats.betaprime.pdf(y, np.exp(param[0]), np.exp(param[1]),
                                                 loc=param[2], scale=np.exp(param[3]))


        elif family =='binom':  ## [R+] ;     support [N+]

                f =  stats.binom.pmf(y, ntrial, np.exp(param[0]))


        elif family =='nbinom':  ## [R+] ;     support [N+]

                f =  stats.nbinom.pmf(y, ntrial, np.exp(param[0]))


        elif family =='bradford':  ## [R+] ;     support [0, 1]

                f =  stats.bradford.pdf(y, np.exp(param[0]), loc=param[1],
                                                scale=np.exp(param[2]))


        elif family =='burr':  ## [R+, R+] ;     support [R+]

                f =  stats.burr.pdf(y, np.exp(param[0]), np.exp(param[1]),
                                            loc=param[2], scale=np.exp(param[3]))


        elif family =='burr12':  ## [R+, R+] ;     support [R+]

                f =  stats.burr12.pdf(y, np.exp(param[0]), np.exp(param[1]),
                                              loc=param[2], scale=np.exp(param[3]))


        elif family =='chi':  ## [R+] ;     support [R+]

                f =  stats.chi.pdf(y, np.exp(param[0]), loc=param[1],
                                           scale=np.exp(param[2]))


        elif family =='chi2':  ## [R+] ;     support [R+]

                f =  stats.chi2.pdf(y, np.exp(param[0]), loc=param[1],
                                            scale=np.exp(param[2]))




        elif family =='dgamma':  ## [R+] ;     support [R]

                f =  stats.dgamma.pdf(y, np.exp(param[0]), loc=param[1],
                                              scale=np.exp(param[2]))


        elif family =='dweibull':  ## [R+] ;     support [R]

                f =  stats.dweibull.pdf(y, np.exp(param[0]), loc=param[1],
                                                scale=np.exp(param[2]))


        elif family =='ev':  ##  []   support [R]

                f =  stats.genextreme.pdf(y, loc=param[0], scale=np.exp(param[1]))


        elif family =='expon':  ## [R+] ;     support [R+]

                f =  stats.expon.pdf(y, loc=param[0], scale=np.exp(param[1]))


        elif family =='exponnorm':  ## [R+] ;     support [R]

                f =  stats.exponnorm.pdf(y, np.exp(param[0]), loc=param[1],
                                                 scale=np.exp(param[2]))


        elif family =='exponweib':  ## [R+, R+] ;     support [R+]

                f =  stats.exponweib.pdf(y, np.exp(param[0]), np.exp(param[1]), loc=param[2],
                                                 scale=np.exp(param[3]))


        elif family =='exponpow':  ## [R+] ;     support [R+]

                f =  stats.exponpow.pdf(y, np.exp(param[0]), loc=param[1],
                                                scale=np.exp(param[2]))


        elif family =='f':  ## [R+, R+] ;     support [R+]

                f =  stats.f.pdf(y, np.exp(param[0]), np.exp(param[1]), loc=param[2],
                                         scale=np.exp(param[3]))


        elif family =='fatiguelife':  ## [R+] ;     support [R+]

                f =  stats.fatiguelife.pdf(y, np.exp(param[0]), loc=param[1],
                                                   scale=np.exp(param[2]))


        elif family =='fisk':  ## [R+] ;     support [R+]

                f =  stats.fisk.pdf(y, np.exp(param[0]), loc=param[1],
                                            scale=np.exp(param[2]))


        elif family =='gamma':  ##  [R+]   support [R+]

                f =  stats.gamma.pdf(y, np.exp(param[0]), loc=param[1],
                                             scale=np.exp(param[2]))


        elif family =='genlogistic':  ## [R+]   support [R+]

                f =  stats.genlogistic.pdf(y, np.exp(param[0]), loc=param[1],
                                                   scale=np.exp(param[2]))


        elif family =='gennorm':  ##  [R+]   support [R]

                f =  stats.gennorm.pdf(y, np.exp(param[0]), loc=param[1],
                                               scale=np.exp(param[2]))

        elif family =='genexpon':  ##  [R+, R+, R+]   support [R+]

                f =  stats.genexpon.pdf(y, np.exp(param[0]), np.exp(param[1]), np.exp(param[2]),
                                                loc=param[3], scale=np.exp(param[4]))


        elif family =='gengamma':  ##  [R+, R+]   support [R+]

                f =  stats.gengamma.pdf(y, np.exp(param[0]), np.exp(param[1]),
                                                loc=param[2], scale=np.exp(param[2]))


        elif family =='geninvgauss':  ##  [R, R+]   support [R+]

                f =  stats.geninvgauss.pdf(y, param[0], np.exp(param[1]),
                                                   loc=param[2], scale=np.exp(param[2]))

        elif family =='geom':  ##  [R+]   support [N+]

                f = stats.geom.pmf(y, np.exp(param[0]))


        elif family =='gompertz':  ##  [R+]   support [R+]

                f =  stats.gompertz.pdf(y, np.exp(param[0]), loc=param[1],
                                                scale=np.exp(param[2]))


        elif family =='gumbel_r':  ##  []   support [R+]

                f =  stats.gumbel_r.pdf(y, loc=param[0], scale=np.exp(param[1]))


        elif family =='gumbel_l':  ##  []   support [R+]

                f =  stats.gumbel_l.pdf(y, loc=param[0], scale=np.exp(param[1]))


        elif family =='invgamma':  ##  [R+]   support [R+]

                f =  stats.invgamma.pdf(y, np.exp(param[0]), loc=param[1],
                                                scale=np.exp(param[2]))


        elif family =='invgauss':  ##  [R+]   support [R+]

                f =  stats.invgauss.pdf(y, np.exp(param[0]), loc=param[1],
                                                scale=np.exp(param[2]))


        elif family =='invweibull':  ##  [R+]   support [R+]

                f =  stats.invweibull.pdf(y, np.exp(param[0]), loc=param[1],
                                                  scale=np.exp(param[2]))


        elif family =='johnsonsu':  ##  [R+, R+]   support [R+]

                f =  stats.johnsonsu.pdf(y, np.exp(param[0]), np.exp(param[1]),
                                                 loc=param[2], scale=np.exp(param[3]))


        elif family =='laplace':  ##  []   support [R]

                f =  stats.laplace.pdf(y, loc=param[0], scale=np.exp(param[1]))


        elif family =='levy':  ##  []   support [R+]

                f =  stats.levy.pdf(y, loc=param[0], scale=np.exp(param[1]))


        elif family =='levy_l':  ##  []   support [R-]

                f =  stats.levy_l.pdf(y, loc=param[0], scale=np.exp(param[1]))


        elif family =='logistic':  ##  []   support [R]

                f =  stats.logistic.pdf(y, loc=param[0], scale=np.exp(param[1]))


        elif family =='loggamma':  ##  [R+]   support [R+]

                f =  stats.loggamma.pdf(y, np.exp(param[0]), loc=param[1],
                                                scale=np.exp(param[2]))


        elif family =='loglaplace':  ##  [R+]   support [R+]

                f =  stats.loglaplace.pdf(y, np.exp(param[0]), loc=param[1],
                                                  scale=np.exp(param[2]))


        elif family =='lognorm':  ##  [R+]   support [R+]

                f =  stats.loglaplace.pdf(y, np.exp(param[0]), loc=param[1],
                                                  scale=np.exp(param[2]))


        elif family =='lomax':  ##  [R+]   support [R+]

                f =  stats.lomax.pdf(y, np.exp(param[0]), loc=param[1],
                                             scale=np.exp(param[2]))


        elif family =='maxwell':  ##  []   support [R+]

                f =  stats.maxwell.pdf(y, loc=param[0], scale=np.exp(param[1]))


        elif family =='mielke':  ##  [R+, R+]   support [R+]

                f =  stats.mielke.pdf(y, np.exp(param[0]), np.exp(param[1]),
                                              loc=param[2], scale=np.exp(param[3]))


        elif family =='moyal':  ##  []   support [R]

                f =  stats.moyal.pdf(y, loc=param[0], scale=np.exp(param[1]))


        elif family =='nakagami':  ##  [R+]   support [R+]

                f =  stats.nakagami.pdf(y, np.exp(param[0]), loc=param[1],
                                                scale=np.exp(param[2]))


        elif family =='ncf':  ##  [R+, R+, R]   support [R+]

                f =  stats.ncf.pdf(y, np.exp(param[0]), np.exp(param[1]), param[2],
                                           loc=param[3], scale=np.exp(param[4]))


        elif family =='nct':  ##  [R+, R]   support [R+]

                f =  stats.nct.pdf(y, np.exp(param[0]), param[1],
                                           loc=param[2], scale=np.exp(param[3]))


        elif family =='ncx2':  ##  [R+, R]   support [R+]

                f =  stats.ncx2.pdf(y, np.exp(param[0]), param[1],
                                            loc=param[2], scale=np.exp(param[3]))


        elif family == 'norm':   ## [] ;     support [R]

                f =  stats.norm.pdf(y, loc=param[0], scale=np.exp(param[1]))



        elif family == 'norminvgauss':   ## [R+, abs(b)>a] ;     support [R]

                f =  stats.norminvgauss.pdf(y, np.exp(param[0]),
                                            np.exp(param[0]) * ( np.exp(2*param[1])-1 ) / ( np.exp(2*param[1])+1 ),
                                            loc=param[2], scale=np.exp(param[3]))


        elif family =='poisson':  ##  [R+]   support [N+]

                f = stats.poisson.pmf(y, np.exp(param[0]))


        elif family =='powerlaw':  ##  [R+]   support [0,1]

                f =  stats.powerlaw.pdf(y, np.exp(param[0]), loc=param[1],
                                                scale=np.exp(param[2]))


        elif family =='powerlognorm':  ##  [R+, R+]   support [R+]

                f =  stats.powerlognorm.pdf(y, np.exp(param[0]), np.exp(param[1]),
                                                    loc=param[2], scale=np.exp(param[3]))


        elif family =='powernorm':  ##  [R+]   support [R+]

                f =  stats.powernorm.pdf(y, np.exp(param[0]), loc=param[1],
                                                 scale=np.exp(param[2]))


        elif family =='rayleigh':  ##  []   support [R+]

                f =  stats.rayleigh.pdf(y, loc=param[0], scale=np.exp(param[1]))


        elif family =='rice':  ##  [R+]   support [R+]

                f =  stats.rice.pdf(y, np.exp(param[0]), loc=param[1],
                                            scale=np.exp(param[2]))


        elif family =='recipinvgauss':  ##  []   support [R+]

                f =  stats.recipinvgauss.pdf(y, param[0], loc=param[1], scale=np.exp(param[2]))



        elif family =='skewnorm':  ##  [R]   support [R]

                f =  stats.skewnorm.pdf(y, param[0], loc=param[1],
                                                scale=np.exp(param[2]))


        elif family =='t':  ##  [R+]   support [R]

                f =  stats.t.pdf(y, np.exp(param[0]), loc=param[1],
                                         scale=np.exp(param[2]))


        elif family =='tukeylambda':  ##  [R]   support [R]

                f =  stats.tukeylambda.pdf(y, param[0], loc=param[1],
                                                   scale=np.exp(param[2]))



        elif family =='wald':  ##  []   support [R+]

                f =  stats.wald.pdf(y, loc=param[0], scale=np.exp(param[1]))




        #===================================
        return(f)



    ##=============================================================================
    ##=============================================================================
    ##=============================================================================
    ##=============================================================================
    def SimMarkovChain(self, Q,n,eta0):
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




    ##=============================================================================
    ##=============================================================================
    ##=============================================================================
    ##=============================================================================
    def SimHMMGen(self, Q, family, theta, n, ntrial=0):

        r = Q.shape[0]
        MC = self.SimMarkovChain(Q,n,1)

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
                sim[0:n,j] = stats.t.rvs(theta[j,0], loc=theta[j,1],
                                         scale=theta[j,2], size=n)


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




    ##=============================================================================
    ##=============================================================================
    ##=============================================================================
    ##=============================================================================
    def SimUnivGen(self, family, theta, n, ntrial=0):

        if (n>1):

            sim = np.zeros((n,1))

            #===================================
            if family =='alpha':  ## [R+] ;     support [R+]

                    sim[0:n] = stats.alpha.rvs(theta[0], loc=theta[1],
                                                 scale=theta[2], size=n)


            elif family =='argus':  ## [R+] ;     support ]0,1[

                    sim[0:n] = stats.argus.rvs(theta[0], loc=theta[1],
                                                 scale=theta[2], size=n)


            elif family =='beta':  ## [R+, R+] ;     support [0, 1]

                    sim[0:n] = stats.beta.rvs(theta[0], theta[1],
                                                loc=theta[2], scale=theta[3], size=n)


            elif family =='betaprime':  ## [R+, R+] ;     support [R+]

                    sim[0:n] = stats.betaprime.rvs(theta[0], theta[1],
                                                     loc=theta[2], scale=theta[3], size=n)


            elif family =='binom':  ## [R+] ;     support [N+]

                    sim[0:n] = stats.binom.rvs(ntrial, theta[0], size=n)


            elif family =='nbinom':  ## [R+] ;     support [N+]

                    sim[0:n] = stats.nbinom.rvs(ntrial, theta[0], size=n)


            elif family =='bradford':  ## [R+] ;     support [0, 1]

                    sim[0:n] = stats.bradford.rvs(theta[0], loc=theta[1],
                                                    scale=theta[2], size=n)


            elif family =='burr':  ## [R+, R+] ;     support [R+]

                    sim[0:n] = stats.burr.rvs(theta[0], theta[1],
                                                loc=theta[2], scale=theta[3], size=n)


            elif family =='burr12':  ## [R+, R+] ;     support [R+]

                    sim[0:n] = stats.burr12.rvs(theta[0], theta[1],
                                                  loc=theta[2], scale=theta[3], size=n)


            elif family =='chi':  ## [R+] ;     support [R+]

                    sim[0:n] = stats.chi.rvs(theta[0], loc=theta[1],
                                               scale=theta[2], size=n)


            elif family =='chi2':  ## [R+] ;     support [R+]

                    sim[0:n] = stats.chi2.rvs(theta[0], loc=theta[1],
                                                scale=theta[2], size=n)



            elif family =='dgamma':  ## [R+] ;     support [R]

                    sim[0:n] = stats.dgamma.rvs(theta[0], loc=theta[1],
                                                  scale=theta[2], size=n)


            elif family =='dweibull':  ## [R+] ;     support [R]

                    sim[0:n] = stats.dweibull.rvs(theta[0], loc=theta[1],
                                                    scale=theta[2], size=n)


            elif family =='ev':  ##  []   support [R]

                    sim[0:n] = stats.genextreme.rvs(loc=theta[0], scale=theta[1], size=n)


            elif family =='expon':  ## [] ;     support [R+]

                    sim[0:n] = stats.expon.rvs(loc=theta[0], scale=theta[1],
                                                 size=n)


            elif family =='exponnorm':  ## [R+] ;     support [R]

                    sim[0:n] = stats.exponnorm.rvs(theta[0], loc=theta[1],
                                                     scale=theta[2], size=n)


            elif family =='exponweib':  ## [R+, R+] ;     support [R+]

                    sim[0:n] = stats.exponweib.rvs(theta[0], theta[1], loc=theta[2],
                                                     scale=theta[3], size=n)


            elif family =='exponpow':  ## [R+] ;     support [R+]

                    sim[0:n] = stats.exponpow.rvs(theta[0], loc=theta[1],
                                                    scale=theta[2], size=n)


            elif family =='f':  ## [R+, R+] ;     support [R+]

                    sim[0:n] = stats.f.rvs(theta[0], theta[1], loc=theta[2],
                                             scale=theta[3], size=n)


            elif family =='fatiguelife':  ## [R+] ;     support [R+]

                    sim[0:n] = stats.fatiguelife.rvs(theta[0], loc=theta[1],
                                                       scale=theta[2], size=n)


            elif family =='fisk':  ## [R+] ;     support [R+]

                    sim[0:n] = stats.fisk.rvs(theta[0], loc=theta[1],
                                                scale=theta[2], size=n)


            elif family =='gamma':  ##  [R+]   support [R+]

                    sim[0:n] = stats.gamma.rvs(theta[0], loc=theta[1],
                                                 scale=theta[2], size=n)


            elif family =='genlogistic':  ## [R+]   support [R+]

                    sim[0:n] = stats.genlogistic.rvs(theta[0], loc=theta[1],
                                                       scale=theta[2], size=n)


            elif family =='gennorm':  ##  [R+]   support [R]

                    sim[0:n] = stats.gennorm.rvs(theta[0], loc=theta[1],
                                                   scale=theta[2], size=n)

            elif family =='genexpon':  ##  [R+, R+, R+]   support [R+]

                    sim[0:n] = stats.genexpon.rvs(theta[0], theta[1], theta[2],
                                                    loc=theta[3], scale=theta[4], size=n)


            elif family =='gengamma':  ##  [R+, R+]   support [R+]

                    sim[0:n] = stats.gengamma.rvs(theta[0], theta[1],
                                                    loc=theta[2], scale=theta[2], size=n)


            elif family =='geninvgauss':  ##  [R, R+]   support [R+]

                    sim[0:n] = stats.geninvgauss.rvs(theta[0], theta[1],
                                                       loc=theta[2], scale=theta[2], size=n)


            elif family =='geom':  ##  [R+]   support [N+]

                    sim[0:n] = stats.geom.rvs(theta[0], size=n)


            elif family =='gompertz':  ##  [R+]   support [R+]

                    sim[0:n] = stats.gompertz.rvs(theta[0], loc=theta[1],
                                                    scale=theta[2], size=n)


            elif family =='gumbel_r':  ##  []   support [R+]

                    sim[0:n] = stats.gumbel_r.rvs(loc=theta[0], scale=theta[1], size=n)


            elif family =='gumbel_l':  ##  []   support [R+]

                    sim[0:n] = stats.gumbel_l.rvs(loc=theta[0], scale=theta[1], size=n)


            elif family =='invgamma':  ##  [R+]   support [R+]

                    sim[0:n] = stats.invgamma.rvs(theta[0], loc=theta[1],
                                                    scale=theta[2], size=n)


            elif family =='invgauss':  ##  [R+]   support [R+]

                    sim[0:n] = stats.invgauss.rvs(theta[0], loc=theta[1],
                                                    scale=theta[2], size=n)


            elif family =='invweibull':  ##  [R+]   support [R+]

                    sim[0:n] = stats.invweibull.rvs(theta[0], loc=theta[1],
                                                      scale=theta[2], size=n)


            elif family =='johnsonsu':  ##  [R+, R+]   support [R+]

                    sim[0:n] = stats.johnsonsu.rvs(theta[0], theta[1],
                                                     loc=theta[2], scale=theta[3], size=n)


            elif family =='laplace':  ##  []   support [R]

                    sim[0:n] = stats.laplace.rvs(loc=theta[0], scale=theta[1], size=n)


            elif family =='levy':  ##  []   support [R+]

                    sim[0:n] = stats.levy.rvs(loc=theta[0], scale=theta[1], size=n)


            elif family =='levy_l':  ##  []   support [R-]

                    sim[0:n] = stats.levy_l.rvs(loc=theta[0], scale=theta[1], size=n)


            elif family =='logistic':  ##  []   support [R]

                    sim[0:n] = stats.logistic.rvs(loc=theta[0], scale=theta[1], size=n)


            elif family =='loggamma':  ##  [R+]   support [R+]

                    sim[0:n] = stats.loggamma.rvs(theta[0], loc=theta[1],
                                                    scale=theta[2], size=n)


            elif family =='loglaplace':  ##  [R+]   support [R+]

                    sim[0:n] = stats.loglaplace.rvs(theta[0], loc=theta[1],
                                                      scale=theta[2], size=n)


            elif family =='lognorm':  ##  [R+]   support [R+]

                    sim[0:n] = stats.loglaplace.rvs(theta[0], loc=theta[1],
                                                      scale=theta[2], size=n)


            elif family =='lomax':  ##  [R+]   support [R+]

                    sim[0:n] = stats.lomax.rvs(theta[0], loc=theta[1],
                                                 scale=theta[2], size=n)


            elif family =='maxwell':  ##  []   support [R+]

                    sim[0:n] = stats.maxwell.rvs(loc=theta[0], scale=theta[1], size=n)


            elif family =='mielke':  ##  [R+, R+]   support [R+]

                    sim[0:n] = stats.mielke.rvs(theta[0], theta[1],
                                                  loc=theta[2], scale=theta[3], size=n)


            elif family =='moyal':  ##  []   support [R]

                    sim[0:n] = stats.moyal.rvs(loc=theta[0], scale=theta[1], size=n)


            elif family =='nakagami':  ##  [R+]   support [R+]

                    sim[0:n] = stats.nakagami.rvs(theta[0], loc=theta[1],
                                                    scale=theta[2], size=n)


            elif family =='ncf':  ##  [R+, R+, R]   support [R+]

                    sim[0:n] = stats.ncf.rvs(theta[0], theta[1], theta[2],
                                               loc=theta[3], scale=theta[4], size=n)


            elif family =='nct':  ##  [R+, R]   support [R+]

                    sim[0:n] = stats.nct.rvs(theta[0], theta[1],
                                               loc=theta[2], scale=theta[3], size=n)


            elif family =='ncx2':  ##  [R+, R]   support [R+]

                    sim[0:n] = stats.ncx2.rvs(theta[0], theta[1],
                                                loc=theta[2], scale=theta[3], size=n)


            elif family == 'norm':   ## [] ;     support [R]

                    sim[0:n,0] = stats.norm.rvs(loc=theta[0], scale=theta[1], size=n)



            elif family == 'norminvgauss':   ## [R+, abs(b)>a] ;     support [R]

                    sim[0:n] = stats.norminvgauss.rvs(theta[0], theta[1],
                                                        loc=theta[2], scale=theta[3], size=n)


            elif family =='poisson':  ##  [R+]   support [N+]

                    sim[0:n] = stats.poisson.rvs(theta[0], size=n)


            elif family =='powerlaw':  ##  [R+]   support [0,1]

                    sim[0:n] = stats.powerlaw.rvs(theta[0], loc=theta[1],
                                                    scale=theta[2], size=n)


            elif family =='powerlognorm':  ##  [R+, R+]   support [R+]

                    sim[0:n] = stats.powerlognorm.rvs(theta[0], theta[1],
                                                        loc=theta[2], scale=theta[3], size=n)


            elif family =='powernorm':  ##  [R+]   support [R+]

                    sim[0:n] = stats.powernorm.rvs(theta[0], loc=theta[1],
                                                     scale=theta[2], size=n)


            elif family =='rayleigh':  ##  []   support [R+]

                    sim[0:n] = stats.rayleigh.rvs(loc=theta[0], scale=theta[1], size=n)


            elif family =='rice':  ##  [R+]   support [R+]

                    sim[0:n] = stats.rice.rvs(theta[0], loc=theta[1],
                                                scale=theta[2], size=n)


            elif family =='recipinvgauss':  ##  [R]   support [R+]

                    sim[0:n] = stats.recipinvgauss.rvs(theta[0], loc=theta[1], scale=theta[2], size=n)



            elif family =='skewnorm':  ##  [R+]   support [R]

                    sim[0:n] = stats.skewnorm.rvs(theta[0], loc=theta[1],
                                                    scale=theta[2], size=n)


            elif family =='t':  ##  [R+]   support [R]

                    sim[0:n] = stats.t.rvs(theta[0], loc=theta[1],
                                             scale=theta[2], size=n)


            elif family =='tukeylambda':  ##  [R]   support [R]

                    sim[0:n] = stats.tukeylambda.rvs(theta[0], loc=theta[1],
                                                       scale=theta[2], size=n)



            elif family =='wald':  ##  []   support [R+]

                    sim[0:n] = stats.wald.rvs(loc=theta[0], scale=theta[1], size=n)



        elif (n==1):

            #===================================
            if family =='alpha':  ## [R+] ;     support [R+]

                    sim = stats.alpha.rvs(theta[0], loc=theta[1],
                                                 scale=theta[2], size=n)


            elif family =='argus':  ## [R+] ;     support ]0,1[

                    sim = stats.argus.rvs(theta[0], loc=theta[1],
                                                 scale=theta[2], size=n)


            elif family =='beta':  ## [R+, R+] ;     support [0, 1]

                    sim = stats.beta.rvs(theta[0], theta[1],
                                                loc=theta[2], scale=theta[3], size=n)


            elif family =='betaprime':  ## [R+, R+] ;     support [R+]

                    sim = stats.betaprime.rvs(theta[0], theta[1],
                                                     loc=theta[2], scale=theta[3], size=n)


            elif family =='binom':  ## [R+] ;     support [N+]

                    sim = stats.binom.rvs(ntrial, theta[0], size=n)


            elif family =='nbinom':  ## [R+] ;     support [N+]

                    sim = stats.nbinom.rvs(ntrial, theta[0], size=n)


            elif family =='bradford':  ## [R+] ;     support [0, 1]

                    sim = stats.bradford.rvs(theta[0], loc=theta[1],
                                                    scale=theta[2], size=n)


            elif family =='burr':  ## [R+, R+] ;     support [R+]

                    sim = stats.burr.rvs(theta[0], theta[1],
                                                loc=theta[2], scale=theta[3], size=n)


            elif family =='burr12':  ## [R+, R+] ;     support [R+]

                    sim = stats.burr12.rvs(theta[0], theta[1],
                                                  loc=theta[2], scale=theta[3], size=n)


            elif family =='chi':  ## [R+] ;     support [R+]

                    sim = stats.chi.rvs(theta[0], loc=theta[1],
                                               scale=theta[2], size=n)


            elif family =='chi2':  ## [R+] ;     support [R+]

                    sim = stats.chi2.rvs(theta[0], loc=theta[1],
                                                scale=theta[2], size=n)



            elif family =='dgamma':  ## [R+] ;     support [R]

                    sim = stats.dgamma.rvs(theta[0], loc=theta[1],
                                                  scale=theta[2], size=n)


            elif family =='dweibull':  ## [R+] ;     support [R]

                    sim = stats.dweibull.rvs(theta[0], loc=theta[1],
                                                    scale=theta[2], size=n)


            elif family =='ev':  ##  []   support [R]

                    sim = stats.genextreme.rvs(loc=theta[0], scale=theta[1], size=n)


            elif family =='expon':  ## [] ;     support [R+]

                    sim = stats.expon.rvs(loc=theta[0], scale=theta[1],
                                                 size=n)


            elif family =='exponnorm':  ## [R+] ;     support [R]

                    sim = stats.exponnorm.rvs(theta[0], loc=theta[1],
                                                     scale=theta[2], size=n)


            elif family =='exponweib':  ## [R+, R+] ;     support [R+]

                    sim = stats.exponweib.rvs(theta[0], theta[1], loc=theta[2],
                                                     scale=theta[3], size=n)


            elif family =='exponpow':  ## [R+] ;     support [R+]

                    sim = stats.exponpow.rvs(theta[0], loc=theta[1],
                                                    scale=theta[2], size=n)


            elif family =='f':  ## [R+, R+] ;     support [R+]

                    sim = stats.f.rvs(theta[0], theta[1], loc=theta[2],
                                             scale=theta[3], size=n)


            elif family =='fatiguelife':  ## [R+] ;     support [R+]

                    sim = stats.fatiguelife.rvs(theta[0], loc=theta[1],
                                                       scale=theta[2], size=n)


            elif family =='fisk':  ## [R+] ;     support [R+]

                    sim = stats.fisk.rvs(theta[0], loc=theta[1],
                                                scale=theta[2], size=n)


            elif family =='gamma':  ##  [R+]   support [R+]

                    sim = stats.gamma.rvs(theta[0], loc=theta[1],
                                                 scale=theta[2], size=n)


            elif family =='genlogistic':  ## [R+]   support [R+]

                    sim = stats.genlogistic.rvs(theta[0], loc=theta[1],
                                                       scale=theta[2], size=n)


            elif family =='gennorm':  ##  [R+]   support [R]

                    sim = stats.gennorm.rvs(theta[0], loc=theta[1],
                                                   scale=theta[2], size=n)

            elif family =='genexpon':  ##  [R+, R+, R+]   support [R+]

                    sim = stats.genexpon.rvs(theta[0], theta[1], theta[2],
                                                    loc=theta[3], scale=theta[4], size=n)


            elif family =='gengamma':  ##  [R+, R+]   support [R+]

                    sim = stats.gengamma.rvs(theta[0], theta[1],
                                                    loc=theta[2], scale=theta[2], size=n)


            elif family =='geninvgauss':  ##  [R, R+]   support [R+]

                    sim = stats.geninvgauss.rvs(theta[0], theta[1],
                                                       loc=theta[2], scale=theta[2], size=n)


            elif family =='geom':  ##  [R+]   support [N+]

                    sim = stats.geom.rvs(theta[0], size=n)


            elif family =='gompertz':  ##  [R+]   support [R+]

                    sim = stats.gompertz.rvs(theta[0], loc=theta[1],
                                                    scale=theta[2], size=n)


            elif family =='gumbel_r':  ##  []   support [R+]

                    sim = stats.gumbel_r.rvs(loc=theta[0], scale=theta[1], size=n)


            elif family =='gumbel_l':  ##  []   support [R+]

                    sim = stats.gumbel_l.rvs(loc=theta[0], scale=theta[1], size=n)


            elif family =='invgamma':  ##  [R+]   support [R+]

                    sim = stats.invgamma.rvs(theta[0], loc=theta[1],
                                                    scale=theta[2], size=n)


            elif family =='invgauss':  ##  [R+]   support [R+]

                    sim = stats.invgauss.rvs(theta[0], loc=theta[1],
                                                    scale=theta[2], size=n)


            elif family =='invweibull':  ##  [R+]   support [R+]

                    sim = stats.invweibull.rvs(theta[0], loc=theta[1],
                                                      scale=theta[2], size=n)


            elif family =='johnsonsu':  ##  [R+, R+]   support [R+]

                    sim = stats.johnsonsu.rvs(theta[0], theta[1],
                                                     loc=theta[2], scale=theta[3], size=n)


            elif family =='laplace':  ##  []   support [R]

                    sim = stats.laplace.rvs(loc=theta[0], scale=theta[1], size=n)


            elif family =='levy':  ##  []   support [R+]

                    sim = stats.levy.rvs(loc=theta[0], scale=theta[1], size=n)


            elif family =='levy_l':  ##  []   support [R-]

                    sim = stats.levy_l.rvs(loc=theta[0], scale=theta[1], size=n)


            elif family =='logistic':  ##  []   support [R]

                    sim = stats.logistic.rvs(loc=theta[0], scale=theta[1], size=n)


            elif family =='loggamma':  ##  [R+]   support [R+]

                    sim = stats.loggamma.rvs(theta[0], loc=theta[1],
                                                    scale=theta[2], size=n)


            elif family =='loglaplace':  ##  [R+]   support [R+]

                    sim = stats.loglaplace.rvs(theta[0], loc=theta[1],
                                                      scale=theta[2], size=n)


            elif family =='lognorm':  ##  [R+]   support [R+]

                    sim = stats.loglaplace.rvs(theta[0], loc=theta[1],
                                                      scale=theta[2], size=n)


            elif family =='lomax':  ##  [R+]   support [R+]

                    sim = stats.lomax.rvs(theta[0], loc=theta[1],
                                                 scale=theta[2], size=n)


            elif family =='maxwell':  ##  []   support [R+]

                    sim = stats.maxwell.rvs(loc=theta[0], scale=theta[1], size=n)


            elif family =='mielke':  ##  [R+, R+]   support [R+]

                    sim = stats.mielke.rvs(theta[0], theta[1],
                                                  loc=theta[2], scale=theta[3], size=n)


            elif family =='moyal':  ##  []   support [R]

                    sim = stats.moyal.rvs(loc=theta[0], scale=theta[1], size=n)


            elif family =='nakagami':  ##  [R+]   support [R+]

                    sim = stats.nakagami.rvs(theta[0], loc=theta[1],
                                                    scale=theta[2], size=n)


            elif family =='ncf':  ##  [R+, R+, R]   support [R+]

                    sim = stats.ncf.rvs(theta[0], theta[1], theta[2],
                                               loc=theta[3], scale=theta[4], size=n)


            elif family =='nct':  ##  [R+, R]   support [R+]

                    sim = stats.nct.rvs(theta[0], theta[1],
                                               loc=theta[2], scale=theta[3], size=n)


            elif family =='ncx2':  ##  [R+, R]   support [R+]

                    sim = stats.ncx2.rvs(theta[0], theta[1],
                                                loc=theta[2], scale=theta[3], size=n)


            elif family == 'norm':   ## [] ;     support [R]

                    sim = stats.norm.rvs(loc=theta[0], scale=theta[1], size=n)



            elif family == 'norminvgauss':   ## [R+, abs(b)>a] ;     support [R]

                    sim = stats.norminvgauss.rvs(theta[0], theta[1],
                                                        loc=theta[2], scale=theta[3], size=n)


            elif family =='poisson':  ##  [R+]   support [N+]

                    sim = stats.poisson.rvs(theta[0], size=n)


            elif family =='powerlaw':  ##  [R+]   support [0,1]

                    sim = stats.powerlaw.rvs(theta[0], loc=theta[1],
                                                    scale=theta[2], size=n)


            elif family =='powerlognorm':  ##  [R+, R+]   support [R+]

                    sim = stats.powerlognorm.rvs(theta[0], theta[1],
                                                        loc=theta[2], scale=theta[3], size=n)


            elif family =='powernorm':  ##  [R+]   support [R+]

                    sim = stats.powernorm.rvs(theta[0], loc=theta[1],
                                                     scale=theta[2], size=n)


            elif family =='rayleigh':  ##  []   support [R+]

                    sim = stats.rayleigh.rvs(loc=theta[0], scale=theta[1], size=n)


            elif family =='rice':  ##  [R+]   support [R+]

                    sim = stats.rice.rvs(theta[0], loc=theta[1],
                                                scale=theta[2], size=n)


            elif family =='recipinvgauss':  ##  [R]   support [R+]

                    sim = stats.recipinvgauss.rvs(theta[0], loc=theta[1], scale=theta[2], size=n)



            elif family =='skewnorm':  ##  [R+]   support [R]

                    sim = stats.skewnorm.rvs(theta[0], loc=theta[1],
                                                    scale=theta[2], size=n)


            elif family =='t':  ##  [R+]   support [R]

                    sim = stats.t.rvs(theta[0], loc=theta[1],
                                             scale=theta[2], size=n)


            elif family =='tukeylambda':  ##  [R]   support [R]

                    sim = stats.tukeylambda.rvs(theta[0], loc=theta[1],
                                                       scale=theta[2], size=n)



            elif family =='wald':  ##  []   support [R+]

                    sim = stats.wald.rvs(loc=theta[0], scale=theta[1], size=n)



        #===================================
        return(sim)




    ##=============================================================================
    ##=============================================================================
    ##=============================================================================
    ##=============================================================================
    def theta2alpha(self, param,typeofparams):

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



    ##=============================================================================
    ##=============================================================================
    ##=============================================================================
    ##=============================================================================
    def LLEMStep(self, y,family,lambda_EM,theta):

        LL = -sum( np.multiply(lambda_EM, np.squeeze(np.log(self.PDFunc(family, y, theta)), -1) ) )

        return(LL)



    ##=============================================================================
    def EMStep(self, y, family, theta, Q, ntrial, optimizer_alg):

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
            f[0:n,j] = self.PDFunc(family, y, theta[j,0:p], ntrial).reshape(n)
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
                                                        np.squeeze(np.log(self.PDFunc(family, y, thetaa, ntrial)), -1) ) )
            if optimizer_alg == 'Nelder-Mead':
                res = minimize(fun, theta[i,0:p], method='Nelder-Mead')  # 'Nelder-Mead'
            elif optimizer_alg == 'CG':
                res = minimize(fun, theta[i, 0:p], method='CG')
            elif optimizer_alg == 'BFGS':
                res = minimize(fun, theta[i, 0:p], method='BFGS')
            elif optimizer_alg == 'L-BFGS-B':
                res = minimize(fun, theta[i, 0:p], method='L-BFGS-B')
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
    def Sn1d(self, U):

        n = len(U)
        u = np.sort(U)
        t = (-0.5 + np.arange(1,n+1) ) / n

        stat = (1/(12*n)) + sum( (u-t)**2 )

        return(stat)



    ##=============================================================================
    def statsdistr(self, family, param, ntrial=0):


        #===================================
        if family =='alpha':  ## [R+] ;     support [R+]

                mean, var, skew, kurt =  stats.alpha.stats(param[0], loc=param[1],
                                             scale=param[2], moments='mvsk')


        elif family =='argus':  ## [R+] ;     support ]0,1[

                mean, var, skew, kurt =  stats.argus.stats(param[0], loc=param[1],
                                             scale=param[2], moments='mvsk')


        elif family =='beta':  ## [R+, R+] ;     support [0, 1]

                mean, var, skew, kurt =  stats.beta.stats(param[0], param[1],
                                            loc=param[2], scale=param[3], moments='mvsk')


        elif family =='betaprime':  ## [R+, R+] ;     support [R+]

                mean, var, skew, kurt =  stats.betaprime.stats(param[0], param[1],
                                                 loc=param[2], scale=param[3], moments='mvsk')


        elif family =='binom':  ## [R+] ;     support [N+]

                mean, var, skew, kurt =  stats.binom.stats(ntrial, param[0], moments='mvsk')


        elif family =='nbinom':  ## [R+] ;     support [N+]

                mean, var, skew, kurt =  stats.nbinom.stats(ntrial, param[0], moments='mvsk')


        elif family =='bradford':  ## [R+] ;     support [0, 1]

                mean, var, skew, kurt =  stats.bradford.stats(param[0], loc=param[1],
                                                scale=param[2], moments='mvsk')


        elif family =='burr':  ## [R+, R+] ;     support [R+]

                mean, var, skew, kurt =  stats.burr.stats(param[0], param[1],
                                            loc=param[2], scale=param[3], moments='mvsk')


        elif family =='burr12':  ## [R+, R+] ;     support [R+]

                mean, var, skew, kurt =  stats.burr12.stats(param[0], param[1],
                                              loc=param[2], scale=param[3], moments='mvsk')


        elif family =='chi':  ## [R+] ;     support [R+]

                mean, var, skew, kurt =  stats.chi.stats(param[0], loc=param[1],
                                           scale=param[2], moments='mvsk')


        elif family =='chi2':  ## [R+] ;     support [R+]

                mean, var, skew, kurt =  stats.chi2.stats(param[0], loc=param[1],
                                            scale=param[2], moments='mvsk')



        elif family =='dgamma':  ## [R+] ;     support [R]

                mean, var, skew, kurt =  stats.dgamma.stats(param[0], loc=param[1],
                                              scale=param[2], moments='mvsk')


        elif family =='dweibull':  ## [R+] ;     support [R]

                mean, var, skew, kurt =  stats.dweibull.stats(param[0], loc=param[1],
                                                scale=param[2], moments='mvsk')


        elif family =='ev':  ##  []   support [R]

                mean, var, skew, kurt =  stats.genextreme.stats(loc=param[0], scale=param[1], moments='mvsk')


        elif family =='expon':  ## [R+] ;     support [R+]

                mean, var, skew, kurt =  stats.expon.stats(loc=param[0], scale=param[1], moments='mvsk')


        elif family =='exponnorm':  ## [R+] ;     support [R]

                mean, var, skew, kurt =  stats.exponnorm.stats(param[0], loc=param[1],
                                                 scale=param[2], moments='mvsk')


        elif family =='exponweib':  ## [R+, R+] ;     support [R+]

                mean, var, skew, kurt =  stats.exponweib.stats(param[0], param[1], loc=param[2],
                                                 scale=param[3], moments='mvsk')


        elif family =='exponpow':  ## [R+] ;     support [R+]

                mean, var, skew, kurt =  stats.exponpow.stats(param[0], loc=param[1],
                                                scale=param[2], moments='mvsk')


        elif family =='f':  ## [R+, R+] ;     support [R+]

                mean, var, skew, kurt =  stats.f.stats(param[0], param[1], loc=param[2],
                                         scale=param[3], moments='mvsk')


        elif family =='fatiguelife':  ## [R+] ;     support [R+]

                mean, var, skew, kurt =  stats.fatiguelife.stats(param[0], loc=param[1],
                                                   scale=param[2], moments='mvsk')


        elif family =='fisk':  ## [R+] ;     support [R+]

                mean, var, skew, kurt =  stats.fisk.stats(param[0], loc=param[1],
                                            scale=param[2], moments='mvsk')


        elif family =='gamma':  ##  [R+]   support [R+]

                mean, var, skew, kurt =  stats.gamma.stats(param[0], loc=param[1],
                                             scale=param[2], moments='mvsk')


        elif family =='genlogistic':  ## [R+]   support [R+]

                mean, var, skew, kurt =  stats.genlogistic.stats(param[0], loc=param[1],
                                                   scale=param[2], moments='mvsk')


        elif family =='gennorm':  ##  [R+]   support [R]

                mean, var, skew, kurt =  stats.gennorm.stats(param[0], loc=param[1],
                                               scale=param[2], moments='mvsk')

        elif family =='genexpon':  ##  [R+, R+, R+]   support [R+]

                mean, var, skew, kurt =  stats.genexpon.stats(param[0], param[1], param[2],
                                                loc=param[3], scale=param[4], moments='mvsk')


        elif family =='gengamma':  ##  [R+, R+]   support [R+]

                mean, var, skew, kurt =  stats.gengamma.stats(param[0], param[1],
                                                loc=param[2], scale=param[2], moments='mvsk')


        elif family =='geninvgauss':  ##  [R, R+]   support [R+]

                mean, var, skew, kurt =  stats.geninvgauss.stats(param[0], param[1],
                                                   loc=param[2], scale=param[2], moments='mvsk')

        elif family =='geom':  ##  [R+]   support [N+]

                mean, var, skew, kurt = stats.geom.stats(param[0], moments='mvsk')


        elif family =='gompertz':  ##  [R+]   support [R+]

                mean, var, skew, kurt =  stats.gompertz.stats(param[0], loc=param[1],
                                                scale=param[2], moments='mvsk')


        elif family =='gumbel_r':  ##  []   support [R+]

                mean, var, skew, kurt =  stats.gumbel_r.stats(loc=param[0], scale=param[1], moments='mvsk')


        elif family =='gumbel_l':  ##  []   support [R+]

                mean, var, skew, kurt =  stats.gumbel_l.stats(loc=param[0], scale=param[1], moments='mvsk')


        elif family =='invgamma':  ##  [R+]   support [R+]

                mean, var, skew, kurt =  stats.invgamma.stats(param[0], loc=param[1],
                                                scale=param[2], moments='mvsk')


        elif family =='invgauss':  ##  [R+]   support [R+]

                mean, var, skew, kurt =  stats.invgauss.stats(param[0], loc=param[1],
                                                scale=param[2], moments='mvsk')


        elif family =='invweibull':  ##  [R+]   support [R+]

                mean, var, skew, kurt =  stats.invweibull.stats(param[0], loc=param[1],
                                                  scale=param[2], moments='mvsk')


        elif family =='johnsonsu':  ##  [R+, R+]   support [R+]

                mean, var, skew, kurt =  stats.johnsonsu.stats(param[0], param[1],
                                                 loc=param[2], scale=param[3], moments='mvsk')


        elif family =='laplace':  ##  []   support [R]

                mean, var, skew, kurt =  stats.laplace.stats(loc=param[0], scale=param[1], moments='mvsk')


        elif family =='levy':  ##  []   support [R+]

                mean, var, skew, kurt =  stats.levy.stats(loc=param[0], scale=param[1], moments='mvsk')


        elif family =='levy_l':  ##  []   support [R-]

                mean, var, skew, kurt =  stats.levy_l.stats(loc=param[0], scale=param[1], moments='mvsk')


        elif family =='logistic':  ##  []   support [R]

                mean, var, skew, kurt =  stats.logistic.stats(loc=param[0], scale=param[1], moments='mvsk')


        elif family =='loggamma':  ##  [R+]   support [R+]

                mean, var, skew, kurt =  stats.loggamma.stats(param[0], loc=param[1],
                                                scale=param[2], moments='mvsk')


        elif family =='loglaplace':  ##  [R+]   support [R+]

                mean, var, skew, kurt =  stats.loglaplace.stats(param[0], loc=param[1],
                                                  scale=param[2], moments='mvsk')


        elif family =='lognorm':  ##  [R+]   support [R+]

                mean, var, skew, kurt =  stats.loglaplace.stats(param[0], loc=param[1],
                                                  scale=param[2], moments='mvsk')


        elif family =='lomax':  ##  [R+]   support [R+]

                mean, var, skew, kurt =  stats.lomax.stats(param[0], loc=param[1],
                                             scale=param[2], moments='mvsk')


        elif family =='maxwell':  ##  []   support [R+]

                mean, var, skew, kurt =  stats.maxwell.stats(loc=param[0], scale=param[1], moments='mvsk')


        elif family =='mielke':  ##  [R+, R+]   support [R+]

                mean, var, skew, kurt =  stats.mielke.stats(param[0], param[1],
                                              loc=param[2], scale=param[3], moments='mvsk')


        elif family =='moyal':  ##  []   support [R]

                mean, var, skew, kurt =  stats.moyal.stats(loc=param[0], scale=param[1], moments='mvsk')


        elif family =='nakagami':  ##  [R+]   support [R+]

                mean, var, skew, kurt =  stats.nakagami.stats(param[0], loc=param[1],
                                                scale=param[2], moments='mvsk')


        elif family =='ncf':  ##  [R+, R+, R]   support [R+]

                mean, var, skew, kurt =  stats.ncf.stats(param[0], param[1], param[2],
                                           loc=param[3], scale=param[4], moments='mvsk')


        elif family =='nct':  ##  [R+, R]   support [R+]

                mean, var, skew, kurt =  stats.nct.stats(param[0], param[1],
                                           loc=param[2], scale=param[3], moments='mvsk')


        elif family =='ncx2':  ##  [R+, R]   support [R+]

                mean, var, skew, kurt =  stats.ncx2.stats(param[0], param[1],
                                            loc=param[2], scale=param[3], moments='mvsk')


        elif family == 'norm':   ## [] ;     support [R]

                mean, var, skew, kurt =  stats.norm.stats(loc=param[0], scale=param[1], moments='mvsk')



        elif family == 'norminvgauss':   ## [R+, abs(b)>a] ;     support [R]

                mean, var, skew, kurt =  stats.norminvgauss.stats(param[0], param[1],
                                                    loc=param[2], scale=param[3], moments='mvsk')


        elif family =='poisson':  ##  [R+]   support [N+]

                mean, var, skew, kurt = stats.poisson.stats(param[0], moments='mvsk')


        elif family =='powerlaw':  ##  [R+]   support [0,1]

                mean, var, skew, kurt =  stats.powerlaw.stats(param[0], loc=param[1],
                                                scale=param[2], moments='mvsk')


        elif family =='powerlognorm':  ##  [R+, R+]   support [R+]

                mean, var, skew, kurt =  stats.powerlognorm.stats(param[0], param[1],
                                                    loc=param[2], scale=param[3], moments='mvsk')


        elif family =='powernorm':  ##  [R+]   support [R+]

                mean, var, skew, kurt =  stats.powernorm.stats(param[0], loc=param[1],
                                                 scale=param[2], moments='mvsk')


        elif family =='rayleigh':  ##  []   support [R+]

                mean, var, skew, kurt =  stats.rayleigh.stats(loc=param[0], scale=param[1], moments='mvsk')


        elif family =='rice':  ##  [R+]   support [R+]

                mean, var, skew, kurt =  stats.rice.stats(param[0], loc=param[1],
                                            scale=param[2], moments='mvsk')


        elif family =='recipinvgauss':  ##  []   support [R+]

                mean, var, skew, kurt =  stats.recipinvgauss.stats(param[0], loc=param[1], scale=param[2], moments='mvsk')



        elif family =='skewnorm':  ##  [R+]   support [R]

                mean, var, skew, kurt =  stats.skewnorm.stats(param[0], loc=param[1],
                                                scale=param[2], moments='mvsk')


        elif family =='t':  ##  [R+]   support [R]

                mean, var, skew, kurt =  stats.t.stats(param[0], loc=param[1],
                                         scale=param[2], moments='mvsk')


        elif family =='tukeylambda':  ##  [R]   support [R]

                mean, var, skew, kurt =  stats.tukeylambda.stats(param[0], loc=param[1],
                                                   scale=param[2], moments='mvsk')



        elif family =='wald':  ##  []   support [R+]

                mean, var, skew, kurt =  stats.wald.stats(loc=param[0], scale=param[1], moments='mvsk')




        #===================================
        return(mean, var, skew, kurt)


    ##=============================================================================
    def EstHMMGen(self, y, reg, family, percentiles=None, max_iter=10000, ninit=300, eps=10e-20, ntrial=0,
                  optimizer_alg='Nelder-Mead', init_rand=False):
        ## init_rand : if True, inialize the parameters of the HMM with 0.8% of the observations
        ## percentiles : a list that contains splits. Used to get initial parameters for each regime
        #               example : [0.5] ==> [0, 0.5] + ]0.5, 1]
        #                         [0.3, 0.7] ==> [0, 0.3] + ]0.3, 0.7] + ]0.7, 1]

        if isinstance(percentiles, list):
            reg = len(percentiles) + 1
            percentiles = [1e-9] + percentiles + [100]


        n = len(y)

        n0 = math.floor(n/reg)
        ind0 = np.arange(n0)

        discreteFam = ['poisson', 'binom', 'geom', 'nbinom']

        p, typeofparams = self.infodistr(family)
        theta0 = np.zeros((reg,p))
        alpha0 = np.zeros((reg,p))

        for j in range(reg):
            if percentiles == None or isinstance(percentiles, list) == False:
                ind = j*n0+ind0
                if init_rand==False:
                    x = y[ind]
                else:
                    x = y[np.random.choice(n, int(np.round(0.8*n)), replace=False)]
                if family not in discreteFam:
                    x = np.append(x,min(y))
                    x = np.append(x,max(y))
                tempFit = self.fitdistr(family, x, ntrial)

            else:
                x = y[(y > np.percentile(y, percentiles[j])) & (y <= np.percentile(y, percentiles[j+1]))]
                tempFit = self.fitdistr(family, x, ntrial)

            theta0[j, 0:p] = tempFit
            alpha0[j, 0:p] = self.theta2alpha(tempFit, typeofparams)

        Q0 = np.ones((reg, reg))/reg

        for k in range(ninit):
            nu_EM, alpha_new_EM, Qnew_EM, eta_EM, eta_bar_EM, lambda_EM, Lambda_EM, LL =\
                self.EMStep(y, family, alpha0, Q0, ntrial, optimizer_alg)
            Q0 = Qnew_EM
            alpha0 = alpha_new_EM
            #print(alpha0)
            #print(Q0)
            #print(k)


        for k in range(max_iter):
            nu_EM, alpha_new_EM, Qnew_EM, eta_EM, eta_bar_EM, lambda_EM, Lambda_EM, LL = self.EMStep(
                y, family, alpha0, Q0, ntrial, optimizer_alg)

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
            theta[j,0:p] = self.alpha2theta(alpha[j,0:p],typeofparams)

        t_mean_s = np.zeros((reg))
        for j in range(reg):
            t_mean_s[j], _, _, _ = self.statsdistr(family, theta[j, :])
        order = t_mean_s.argsort()
        theta = theta[order, :]
        temp_Q = Q[order,:]
        Q = temp_Q[:,order]
        eta_EM = eta_EM[:, order]
        lambda_EM = lambda_EM[:, order]
        # Lambda_EM = Lambda_EM[:, order]

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
                cdf_gof[0:n,j] = np.multiply((1-u_Ros), np.squeeze(self.CDF(family, y-0.7, theta[j,0:p],ntrial),-1)
                                             ) +  np.multiply(u_Ros, np.squeeze(self.CDF(
                    family, y, theta[j,0:p],ntrial),-1))
        else:
            for j in range(reg):
                cdf_gof[0:n,j] = np.squeeze(self.CDF(family, y, theta[j,0:p]),-1)


        eta00 = np.ones((1,reg))/reg
        w00 = np.concatenate((eta00, eta_EM), axis=0)
        W = w00[0:n,0:reg].dot(Q)
        U = np.sum( np.multiply(W, cdf_gof), -1 )
        cvm = self.Sn1d(U)

        pred_e = np.zeros((n,1))
        pred_l = np.zeros((n,1))
        for i in range(n):
            bb_e = np.argsort(eta_EM[i,0:reg])
            pred_e[i] = bb_e[-1]
            bb_l = np.argsort(lambda_EM[i,0:reg])
            pred_l[i] = bb_l[-1]
        pred_e = pred_e+1
        pred_l = pred_l+1

        statistics = np.zeros((reg,4))
        mean_s = np.zeros((reg,1))
        var_s = np.zeros((reg,1))
        skew_s = np.zeros((reg,1))
        kurt_s = np.zeros((reg,1))
        for j in range(reg):
            mean_s[j], var_s[j], skew_s[j], kurt_s[j] = self.statsdistr(family, theta[j,0:p])

        statistics[0:reg,0] = mean_s.squeeze(-1)
        statistics[0:reg,1] = var_s.squeeze(-1)**0.5
        statistics[0:reg,2] = skew_s.squeeze(-1)
        statistics[0:reg,3] = kurt_s.squeeze(-1)

        out = {}
        out['theta'] = theta
        out['Q'] = Q
        out['eta_EM'] = eta_EM
        out['nu_EM'] = nu_EM
        out['U'] = U
        out['cvm'] = cvm
        out['W'] = W
        out['lambda_EM'] = lambda_EM
        out['LL'] = LL
        out['AIC'] = AIC
        out['BIC'] = BIC
        out['CAIC'] = CAIC
        out['AICc'] = AICc
        out['HQC'] = HQC
        out['pred_e'] = pred_e
        out['pred_l'] = pred_l
        out['statistics'] = statistics

        return out



    ##=============================================================================
    ##=============================================================================
    ##=============================================================================
    ##=============================================================================
    def bootstrapfun(self, n, family, Q, theta, percentiles, max_iter=10000, eps=10e-4, ntrial=0):

        y1, sim, MC = self.SimHMMGen(Q=Q, family=family, theta=theta, n=int(n), ntrial=ntrial)

        reg = theta.shape[0]

        out = self.EstHMMGen(y=y1, reg=reg, family=family, percentiles=percentiles,
                             max_iter=max_iter, eps=eps, ntrial=ntrial)

        return out['cvm']

    ##=============================================================================
    ##=============================================================================
    ##=============================================================================
    ##=============================================================================

    def GofHMMGen(self, y, reg, family, percentiles=None, max_iter=10000, eps=10e-4, B=100, ntrial=0):

        out = self.EstHMMGen(y=y, reg=reg, family=family, percentiles=percentiles, max_iter=max_iter, eps=eps,
                             ntrial=ntrial)

        cvm_sim = np.zeros((B,1))
        n = len(y)

        n_args = np.ones(B)*n
        n_args = [int(val) for val in n_args]

        print("First estimation done, cvm = ", out['cvm'])

        num_cores = multiprocessing.cpu_count()
        cvm_sim = Parallel(n_jobs=num_cores, verbose=10)(delayed(self.bootstrapfun)(
            int(i), family, out['Q'], out['theta'], percentiles, max_iter, eps, ntrial) for i in n_args)
        pvalue = 100*np.mean(cvm_sim>out['cvm'])

        out['pvalue'] = pvalue

        return out


    def ForecastHMMeta(self, ynew, family, theta, Q, eta):

        r = Q.shape[0]
        p = theta.shape[1]
        n = len(ynew)
        etanew = np.zeros((n, r))

        for j in range(r):
            for l in range(r):
                etanew[0:n,j] = etanew[0:n,j] + eta[l] * Q[l,j]
            etanew[0:n,j] = etanew[0:n,j] * self.PDF(family, ynew, theta[j,0:p])    ## numerateur

        etanew = etanew/etanew.sum(axis=1)[:,None]

        return(etanew)


    def ForecastHMMeta_rolling(self, ynew_s, family, theta, Q, eta_initial):

        r = Q.shape[0]
        n = len(ynew_s)
        if n<=1:
            Warning("length of the observations should be greater than 1")

        etanew_s = np.zeros((n, r))
        etanew_s[0,0:r] = self.ForecastHMMeta(ynew_s[0], family, theta, Q, eta_initial)
        for i in range(1, n):
            etanew_s[i,0:r] = self.ForecastHMMeta(ynew_s[i], family, theta, Q, etanew_s[i-1,0:r])

        return(etanew_s)


    def ForecastHMMPdf(self, y, family, theta, Q, eta, k=1):

        r = Q.shape[0]
        p = theta.shape[1]
        n = len(y)

        if (type(k) != int):
            pdf = np.zeros((n, len(k)))
            for d in range(len(k)):
                Q_prime = np.linalg.matrix_power(Q, k[d])
                for l in range(r):
                    for j in range(r):
                        pdf[0:n,d] = pdf[0:n,d] + eta[j] * Q_prime[j,l] * self.PDF(family, y, theta[l,0:p])

        elif (type(k) == int):
            pdf = np.zeros((n, 1))
            for d in range(1):
                Q_prime = np.linalg.matrix_power(Q, k)
                for l in range(r):
                    for j in range(r):
                        pdf[0:n,d] = pdf[0:n,d] + eta[j] * Q_prime[j,l] * self.PDF(family, y, theta[l,0:p])

        return(pdf)


    def ForecastHMMCdf(self, y, family, theta, Q, eta, k=1):

        r = Q.shape[0]
        p = theta.shape[1]
        n = len(y)

        if (type(k) != int):
            cdf = np.zeros((n, len(k)))
            for d in range(len(k)):
                Q_prime = np.linalg.matrix_power(Q, k[d])
                for l in range(r):
                    for j in range(r):
                        cdf[0:n,d] = cdf[0:n,d] + eta[j] * Q_prime[j,l] * self.CDF(family, y, theta[l,0:p])

        elif (type(k) == int):
            cdf = np.zeros((n, 1))
            for d in range(1):
                Q_prime = np.linalg.matrix_power(Q, k)
                for l in range(r):
                    for j in range(r):
                        cdf[0:n,d] = cdf[0:n,d] + eta[j] * Q_prime[j,l] * self.CDF(family, y, theta[l,0:p])

        return(cdf)


    def ForecastHMMStatistics(self, family, theta, Q, eta, k=1):

        r = Q.shape[0]
        p = theta.shape[1]
        if (type(k) != int):
            mean_s = np.zeros((len(k),1))
            var_s = np.zeros((len(k),1))
            skew_s = np.zeros((len(k),1))
            kurt_s = np.zeros((len(k),1))

            for d in range(len(k)):
                Q_prime = np.linalg.matrix_power(Q, k[d])
                for l in range(r):
                    for j in range(r):
                        mean, var, skew, kurt = self.statsdistr(family, theta[l,0:p])
                        mean_s[d] = mean_s[d] + eta[j] * Q_prime[j,l] * mean
                        var_s[d] = var_s[d] + eta[j] * Q_prime[j,l] * var
                        skew_s[d] = skew_s[d] + eta[j] * Q_prime[j,l] * skew
                        kurt_s[d] = kurt_s[d] + eta[j] * Q_prime[j,l] * kurt

        elif (type(k) == int):
            mean_s = 0
            var_s = 0
            skew_s = 0
            kurt_s = 0

            Q_prime = np.linalg.matrix_power(Q, k)
            for l in range(r):
                for j in range(r):
                    mean, var, skew, kurt = self.statsdistr(family, theta[l,0:p])
                    mean_s = mean_s + eta[j] * Q_prime[j,l] * mean
                    var_s = var_s + eta[j] * Q_prime[j,l] * var
                    skew_s = skew_s + eta[j] * Q_prime[j,l] * skew
                    kurt_s = kurt_s + eta[j] * Q_prime[j,l] * kurt

        return (mean_s, var_s, skew_s, kurt_s)


    def RollingForecastHMM(self, ynew_s, family, theta, Q, eta_initial):

        r = Q.shape[0]
        n = len(ynew_s)
        y_hat = np.zeros((n,1))
        etanew_s = self.ForecastHMMeta_rolling(ynew_s, family, theta, Q, eta_initial)
        y_hat[0], __, __, __, = self.ForecastHMMStatistics(family, theta, Q, eta_initial)

        for i in range(1,n):
            y_hat[i], __, __, __, = self.ForecastHMMStatistics(family, theta, Q, etanew_s[i-1])

        return(y_hat)


    def ForecastHMMRandom(self, family, theta, Q, eta, k=1):

        r = Q.shape[0]
        p = theta.shape[1]
        Q_prime = np.linalg.matrix_power(Q, k)
        y_sample = 0
        for l in range(r):
            for j in range(r):
                y_sample = y_sample + eta[j] * Q_prime[j,l] * self.SimUnivGen(family, theta[l,0:p], 1)

        return(y_sample)


    def RollingForecastHMM_byCum(self, ynew_s, family, theta, Q, eta_initial):

        r = Q.shape[0]
        n = len(ynew_s)
        y_hat = np.zeros((n,1))
        etanew_s = self.ForecastHMMeta_rolling(ynew_s, family, theta, Q, eta_initial)
        y_hat[0] = self.ForecastHMMRandom(family, theta, Q, eta_initial)

        for i in range(1,n):
            y_hat[i] = self.ForecastHMMRandom(family, theta, Q, etanew_s[i-1])

        return(y_hat)


    def RollingForecastHMM_pred_e(self, ynew_s, family, theta, Q, eta_initial):

        r = Q.shape[0]
        n = len(ynew_s)
        y_hat = np.zeros((n,1))
        p = theta.shape[1]

        etanew_s = self.ForecastHMMeta_rolling(ynew_s, family, theta, Q, eta_initial)

        pred_e = np.zeros((n,1))
        for i in range(n):
            bb_e = np.argsort(etanew_s[i,0:r])
            pred_e[i] = bb_e[-1]

        for i in range(n):
            l = int(pred_e[i])
            y_hat[i] = self.SimUnivGen(family, theta[l,0:p], 1)

        return(y_hat)




