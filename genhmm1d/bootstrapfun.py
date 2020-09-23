# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 03:57:56 2020

@author: 49009427
"""

import scipy as sp
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.optimize import minimize
import pandas as pd
from functools import partial
import multiprocessing  

from SimHMMGen import SimHMMGen
from EstHMMGen import EstHMMGen


def bootstrapfun(n, family, Q, theta, max_iter, eps, ntrial=0):
    
    y1, sim, MC = SimHMMGen(Q, family, theta, int(n), ntrial)
    
    reg = theta.shape[0]

    theta1, Q1, eta1, nu1, U1, cvm1, W1, lambda_EM1, LL1, AIC1, BIC1, CAIC1, AICc1, HQC1 = EstHMMGen(
        y1, reg, family, max_iter, eps, ntrial)
    
    return(cvm1)