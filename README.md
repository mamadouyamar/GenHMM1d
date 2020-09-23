# General 1d HMM models

Inference, goodness-of-fit tests, and predictions for continuous and discrete univariate Hidden Markov Models (HMM). The goodness-of-fit test is based on a Cramer von Mises statistic and uses parametric bootstrap to estimate the p-value. The description of the methodology is taken from Nasri et al (2020) <doi: 10.1029/2019WR025122>


## Installation

To install GenHMM1d simply run 
```sh
$ pip install git+https://github.com/mamadouyamar/GenHMM1d.git
```


## Requirements
GenHMM1d requires the following libraries 
	- scipy 
	- matplotlib.pyplot 
	- numpy
	- math
	- scipy 
	- joblib
	- multiprocessing
 
 When unavailable on your system, each of these package can be installed with the following command

```sh
$ pip install package_name
```

## Usage

Import the needed libraries for this example 

```sh
$ import scipy as sp
$ import matplotlib.pyplot as plt
$ import numpy as np
$ from GenHMM1d import hmm 
```








