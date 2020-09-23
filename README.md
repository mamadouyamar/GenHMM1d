# General 1d HMM models

Inference, goodness-of-fit tests, and predictions for continuous and discrete univariate Hidden Markov Models (HMM). The goodness-of-fit test is based on a Cramer von Mises statistic and uses parametric bootstrap to estimate the p-value. The description of the methodology is taken from Nasri et al (2020) <doi: 10.1029/2019WR025122>


## Installation

To install GenHMM1d simply run 
```sh
pip install git+https://github.com/mamadouyamar/GenHMM1d.git
```


## Requirements
GenHMM1d requires the following libraries 
* scipy 
* matplotlib.pyplot 
* numpy
* math
* scipy 
* joblib
* multiprocessing
 
 When unavailable on your system, each of these packages can be installed with the following command

```sh
pip install package_name
```

## Usage

Import the needed libraries for this example 

```sh
import scipy as sp
import matplotlib.pyplot as plt
import numpy as np
from GenHMM1d import hmm 
```

**To generate observations from a particular model, one need to specify the following quantities**

* Q ==> transition matrix
* family ==> the name of the univariate distribution, see documentation (https://docs.scipy.org/doc/scipy/reference/stats.html)
* n ==> number of observations to generate
* ntrial ==> only for the binom and nbinom families

For example one could generate observations from the norm, laplace and binom families with 

```sh
Q = np.zeros((2,2))
Q[0,0] = 0.7
Q[0,1] = 0.3
Q[1,0] = 0.2
Q[1,1] = 0.8

n = 1000

family = 'binom'
ntrial = 5
theta_sim = np.array([[0.5],[0.75]])

y_binom, __, __ = hmm.SimHMMGen(Q, family, theta_sim, n, ntrial)
plt.plot(y_binom)
plt.show()


family = 'laplace'
theta_sim = np.array([[-0.5, 0.7],[0.2, 2.4]])

y_laplace, __, __ = hmm.SimHMMGen(Q, family, theta_sim, n)
plt.plot(y_laplace)
plt.show()


family = 'norm'
theta_sim = np.array([[-0.5, 0.7],[0.2, 2.4]])

y_norm, __, __ = hmm.SimHMMGen(Q, family, theta_sim, n)
plt.plot(y_norm)
plt.show()

```


**Given the previously simulated serie y_norm, one could fit a two regimes HMM with **

```sh
reg = 2  
family = 'norm' 
theta, Q, eta_EM, nu_EM, U, cvm, W, lambda_EM, LL, AIC, BIC, CAIC, AICc, HQC = hmm.EstHMMGen(y_norm, reg, family)
print('theta = ', theta)
print('Q = ', Q)
print('AIC = ', AIC)
print('BIC = ', BIC)
print('cvm = ', cvm)
```

**One could perform a goodness-of-fit test for the two regimes HMM norm with  **

```sh
reg = 2
family = 'norm' 
max_iter = 10000  ## maximum number of iterations of the EM algorithm
eps = 10e-4   ## precision (stopping criteria), suggestion 0.001
B = 100  ## number of bootstap samples
pvalue, Q, theta, eta_EM, cvm, cvm_sim, nu_EM, U, W, AIC, BIC, CAIC, AICc, HQC, LL, lambda_EM = hmm.GofHMMGen(y_norm, reg, family, max_iter, eps, B)

## The model is valid if the pvalue is greater than 5.
print('pvalue = ', pvalue) 
```


**One could perform a goodness-of-fit test for the two regimes HMM binom with  **

```sh
reg = 2
family = 'binom' 
max_iter = 10000  ## maximum number of iterations of the EM algorithm
eps = 10e-4   ## precision (stopping criteria), suggestion 0.001
B = 100  ## number of bootstap samples
ntrial = 5
pvalue, Q, theta, eta_EM, cvm, cvm_sim, nu_EM, U, W, AIC, BIC, CAIC, AICc, HQC, LL, lambda_EM = hmm.GofHMMGen(y_binom, reg, family, max_iter, eps, B, ntrial)

## The model is valid if the pvalue is greater than 5.
print('pvalue = ', pvalue) 
```



