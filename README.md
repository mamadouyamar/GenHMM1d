# GenHMM1d — general univariate and copula Hidden Markov Models

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamadouyamar/GenHMM1d/blob/master/examples.ipynb)

GenHMM1d performs inference, goodness-of-fit testing, and prediction for Hidden
Markov Models. Estimation is by the EM algorithm; the goodness-of-fit test uses
a Cramér–von Mises statistic with parametric bootstrap, following
Nasri et al. (2020) <[doi:10.1029/2019WR025122](https://doi.org/10.1002/cjs.11534)>.

## Model classes

| Model class | Simulate | Estimate | Module |
|---|---|---|---|
| iid HMM, 50+ continuous & discrete families (norm, poisson, t, laplace, binom, …) | `HMM.SimHMMGen` | `HMM.EstHMMGen` | `genhmm1d.hmm` |
| Zero-inflated HMM (a regime with point mass at 0) | `HMM.SimZIHMMGen` | `HMM.EstHMMGen(..., ZI=1)` | `genhmm1d.hmm` |
| AR(1) Gaussian HMM (model M1) | `ARHMM.SimARXHMMGen` | `ARHMM.EstHMMGen_AR` | `genhmm1d.ar_hmm` |
| AR(1) log-linear Poisson HMM (M2) | `ARHMM.SimARPoissonGen` | `ARHMM.EstHMMGen_AR(..., family='poisson')` | `genhmm1d.ar_hmm` |
| AR(1) zero-inflated Gaussian / Poisson HMM (M3, M4) | `ARHMM.SimARZIGaussGen`, `ARHMM.SimARZIPoisGen` | `ARHMM.EstHMMGen_AR(..., ZI=1)` | `genhmm1d.ar_hmm` |
| Regime-switching bivariate copulas (gaussian, t, clayton, frank, gumbel) | `HMMCopula.SimHMMCopula` | `HMMCopula.EstHMMCop` | `genhmm1d.hmm_copula` |

The autoregressive models M1–M4 follow Nasri, Rémillard & Thioub (2024),
*Journal of Statistical Computation and Simulation* (Appendix 1 EM algorithm).
`EstHMMCop` is a Python port of `HMMcopula::EstHMMCop` (R, v1.0.4),
cross-validated against the R implementation.

**Every model class is demonstrated in [`examples.ipynb`](examples.ipynb)** —
each example simulates data with known parameters and estimates them back, so
you can see the recovery quality directly (all examples seeded and
reproducible). Open it in Colab with the badge above.

## Installation

```sh
pip install git+https://github.com/mamadouyamar/GenHMM1d.git
```

Requires Python ≥ 3.6, with numpy, scipy, matplotlib, and joblib.

## Quick start

### iid HMM: simulate, estimate, compare

```python
import numpy as np
from genhmm1d.hmm import HMM

hmm = HMM()
Q = np.array([[0.94, 0.06],
              [0.03, 0.97]])                 # transition matrix
theta = np.array([[0.0,   1.0],
                  [1.349, 1.0]])             # [mu, sd] per regime

np.random.seed(1000)
y, _, _ = hmm.SimHMMGen(Q, 'norm', theta, 5000, burn_in=1000)
out = hmm.EstHMMGen(np.asarray(y).reshape(-1, 1), 2, 'norm')
print(out["theta"], out["Q"])                # also: AIC, BIC, cvm, eta_EM…
```

Output (true → estimated):

```
theta: [[0.0, 1.0], [1.349, 1.0]]  →  [[0.003, 1.000], [1.356, 0.989]]
Q:     [[0.94, 0.06], [0.03, 0.97]]  →  [[0.942, 0.058], [0.039, 0.961]]
```

The `family` argument accepts 150+ scipy.stats distributions
([list](https://github.com/mamadouyamar/GenHMM1d/blob/master/distributions));
discrete families like `'poisson'` and `'binom'` (with `ntrial=`) work the same
way.

### Zero-inflated HMM

Regime 0 is a point mass at zero (e.g., dry days in precipitation series, zero
counts):

```python
theta = np.array([[0.0], [9.0]])             # [lambda]; row 0 = zero regime
np.random.seed(1000)
y, _, _ = hmm.SimZIHMMGen(Q, 'poisson', theta, 5000, burn_in=1000)
out = hmm.EstHMMGen(np.asarray(y).reshape(-1, 1), 2, 'poisson', ZI=1)
```

```
lambda: [0, 9.0]  →  [0, 8.931]
Q:      [[0.94, 0.06], [0.03, 0.97]]  →  [[0.946, 0.054], [0.034, 0.966]]
```

### Autoregressive HMM (models M1–M4)

AR(1) log-linear Poisson HMM (M2), where
`mu_t = exp(alpha_k + phi_k * log(1 + y_(t-1)))` in regime `k`:

```python
from genhmm1d.ar_hmm import ARHMM
arhmm = ARHMM()

theta = np.array([[0.2, 0.3],
                  [1.5, 0.4]])               # [alpha, phi] per regime
np.random.seed(1000)
y, _, _ = arhmm.SimARPoissonGen(Q, theta, 5000, burn_in=1000)
iQ = np.array([[0.9, 0.1], [0.1, 0.9]])     # persistent starting Q, recommended
out = arhmm.EstHMMGen_AR(np.asarray(y).ravel(), 2, family='poisson',
                         p_AR=1, percentiles=[50], initial_Q=iQ)
```

```
[alpha, phi]: [[0.2, 0.3], [1.5, 0.4]]  →  [[0.187, 0.334], [1.544, 0.382]]
Q:            [[0.94, 0.06], [0.03, 0.97]]  →  [[0.945, 0.055], [0.034, 0.966]]
```

`family='norm'` gives the AR(1)-Gaussian model (M1); adding `ZI=1` gives the
zero-inflated AR models (M3 with Gaussian regimes, M4 with Poisson regimes).
See `examples.ipynb` for all four.

### Regime-switching bivariate copulas

```python
from genhmm1d.hmm_copula import HMMCopula
hcop = HMMCopula()

tau = np.array([0.3, 0.7])                   # Kendall tau per regime
np.random.seed(1000)
u, _, _, _ = hcop.SimHMMCopula(Q, 'clayton', tau, 5000, burn_in=1000)
out = hcop.EstHMMCop(u, 2, 'clayton')
```

```
tau: [0.3, 0.7]  →  [0.291, 0.685]
Q:   [[0.94, 0.06], [0.03, 0.97]]  →  [[0.935, 0.065], [0.038, 0.962]]
```

Families: `'gaussian'`, `'t'` (with `DoF=`), `'clayton'`, `'frank'`,
`'gumbel'`.

## Goodness-of-fit test

Cramér–von Mises statistic with parametric bootstrap (B bootstrap samples):

```python
out = hmm.GofHMMGen(y=y, reg=2, family='norm', max_iter=10000, eps=1e-3, B=100)
print(out['pvalue'])   # model not rejected at the 5% level if pvalue > 0.05
```

## Forecasting

Given fitted parameters, forecast regime probabilities and the predictive
density/CDF at horizons `k`:

```python
est = hmm.EstHMMGen(np.asarray(y).reshape(-1, 1), 2, 'norm')
eta = est['eta_EM'][-1, 0:2]

# P(regime | new observation)
probs = hmm.ForecastHMMeta(ynew=np.array([0.5, 1.0]), family='norm',
                           theta=est['theta'], Q=est['Q'], eta=eta)

# predictive pdf / cdf over a grid, horizons k = 1, 2, 5
grid = np.arange(-5, 5, 0.1)
pdf = hmm.ForecastHMMPdf(y=grid, family='norm', theta=est['theta'],
                         Q=est['Q'], eta=eta, k=[1, 2, 5])
cdf = hmm.ForecastHMMCdf(y=grid, family='norm', theta=est['theta'],
                         Q=est['Q'], eta=eta, k=[1, 2, 5])
```

## References

- Nasri, B. R., Rémillard, B. N., & Thioub, M. Y. (2024). Regime-switching
  autoregressive models with hidden and observable regimes. *Journal of
  Statistical Computation and Simulation.* (AR models M1–M4)
- Nasri, B. R., et al. (2020). <[doi:10.1029/2019WR025122](https://doi.org/10.1002/cjs.11534)> (HMM inference and
  Cramér–von Mises goodness-of-fit methodology)

## Contributing

Please report bugs to mamadou.yamar.thioub@hec.ca with:
* a clear and descriptive title;
* the exact steps necessary to reproduce the problem;
* your environment (`pip freeze` output, Python version);
* a minimal code example.

## Contact

Mamadou Yamar Thioub — [@MamadouYamar](https://twitter.com/MamadouYamar) —
mamadou-yamar.thioub@hec.ca

Project link: <https://github.com/mamadouyamar/GenHMM1d>
