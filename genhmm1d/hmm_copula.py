# -*- coding: utf-8 -*-
"""
Python port of HMMcopula::SimHMMCop  (Nasri, Remillard & Thioub, 2020).

Simulation of a bivariate Markov regime-switching copula model:
    - a hidden chain S_1, ..., S_n with transition matrix Q selects, at each t,
      one of `reg` bivariate copulas;
    - within regime k the pair (U_t, V_t) is drawn from a copula whose Kendall
      tau is KendallTau[k]; the tau is mapped to the native copula parameter
      alpha[k] exactly as the R package does (via inverse Kendall tau).

Families: 'gaussian', 't', 'clayton', 'frank', 'gumbel'.
Output margins are uniform on [0,1] (copula pseudo-observations), matching
copula::rCopula in the R code.
"""

import numpy as np
from scipy.optimize import brentq
from scipy.special import spence
from statsmodels.distributions.copula.api import (
    GaussianCopula, StudentTCopula, ClaytonCopula, FrankCopula, GumbelCopula,
)

from .hmm import HMM


class HMMCopula:

    ##=============================================================================
    def tau2alpha(self, family, tau, DoF=None):
        """
        Inverse Kendall tau: map a Kendall's tau to the native copula parameter,
        mirroring copula::iTau in HMMcopula::SimHMMCop.

        :param family: 'gaussian', 't', 'clayton', 'frank', 'gumbel'
        :param tau: Kendall's rank correlation
        :param DoF: degrees of freedom (only used, downstream, for the t copula)
        :return: copula parameter alpha
        """
        if family in ('gaussian', 't'):
            # both use the elliptical relation tau = (2/pi) arcsin(rho)
            alpha = np.sin(np.pi * tau / 2)
        elif family == 'clayton':
            alpha = 2 * tau / (1 - tau)
        elif family == 'gumbel':
            alpha = 1 / (1 - tau)
        elif family == 'frank':
            # no closed form: invert tau(theta) numerically
            # (statsmodels' tau() overflows exp() at large theta -> harmless)
            fun = lambda th: FrankCopula(theta=th, k_dim=2).tau() - tau
            with np.errstate(over='ignore'):
                alpha = brentq(fun, 1e-6, 1e3)
        else:
            raise ValueError("family must be 'gaussian', 't', 'clayton', "
                             "'frank' or 'gumbel'")
        return alpha

    ##=============================================================================
    def _copula(self, family, alpha, DoF=None):
        """Build the statsmodels copula object for parameter alpha."""
        if family in ('gaussian',):
            return GaussianCopula(corr=alpha, k_dim=2)
        elif family == 't':
            return StudentTCopula(corr=alpha, df=DoF, k_dim=2)
        elif family == 'clayton':
            return ClaytonCopula(theta=alpha, k_dim=2)
        elif family == 'frank':
            return FrankCopula(theta=alpha, k_dim=2)
        elif family == 'gumbel':
            return GumbelCopula(theta=alpha, k_dim=2)

    ##=============================================================================
    def SimHMMCopula(self, Q, family, KendallTau, n, DoF=None, burn_in=0):
        """
        Simulate a bivariate Markov regime-switching copula model.

        :param Q: transition matrix (reg x reg)
        :param family: 'gaussian', 't', 'clayton', 'frank', 'gumbel'
        :param KendallTau: Kendall's tau, one per regime (length reg)
        :param n: number of simulated bivariate vectors (after burn-in removal)
        :param DoF: degrees of freedom, only for the Student ('t') copula
        :param burn_in: number of initial observations discarded
        :return: (SimData, MC, alpha)
                 SimData : (n, 2)        selected bivariate uniforms
                 MC      : (n, 1)        hidden chain
                 alpha   : (reg,)        per-regime copula parameters
                 (also returns Sim, the (n, 2*reg) per-regime draws, as a 4th item)
        """
        Q = np.asarray(Q, dtype=float)
        reg = Q.shape[0]
        KendallTau = np.asarray(KendallTau, dtype=float)

        if family == 't' and DoF is None:
            raise ValueError("DoF is required for the Student ('t') copula")

        N = n + burn_in

        # hidden chain (reuse the package's Markov-chain simulator, eta0=1)
        if reg >= 2:
            MC = HMM().SimMarkovChain(Q=Q, n=N, eta0=1)
        else:
            MC = np.zeros((N, 1), dtype=int)

        # tau -> native copula parameter per regime
        alpha = np.zeros(reg)
        for k in range(reg):
            alpha[k] = self.tau2alpha(family, KendallTau[k], DoF)

        # per-regime bivariate draws, stored as columns (2k-1, 2k) as in R.
        # random_state is the global legacy RandomState, so np.random.seed(...)
        # controls reproducibility (consistent with the rest of the package).
        rs = np.random.mtrand._rand
        Sim = np.zeros((N, 2 * reg))
        for k in range(reg):
            u = self._copula(family, alpha[k], DoF).rvs(N, random_state=rs)
            Sim[:, (2 * k):(2 * k + 2)] = u

        # select the active regime's pair at each t
        SimData = np.zeros((N, 2))
        for i in range(N):
            k = int(MC[i][0])
            SimData[i, :] = Sim[i, (2 * k):(2 * k + 2)]

        #=================================== discard burn-in
        SimData = SimData[burn_in:, :]
        Sim     = Sim[burn_in:, :]
        MC      = MC[burn_in:]

        return (SimData, MC, alpha, Sim)


    ##=========================================================================
    ## ESTIMATION SUPPORT (port of HMMcopula::EstHMMCop, estimation only).
    ## Two parameter scales, exactly as the R package:
    ##   theta = native/constrained copula parameter (rho; clayton/gumbel
    ##           theta; frank theta -- the CRAN copula-package Frank parameter)
    ##   alpha = UNCONSTRAINED EM parameter (ParamTau.R / ParamCop.R):
    ##           gaussian/t: alpha = log((1+rho)/(1-rho)),  rho = tanh(alpha/2)
    ##           clayton   : alpha = log(theta),            theta = exp(alpha)
    ##           frank     : alpha = theta (identity)
    ##           gumbel    : alpha = log(theta - 1),        theta = exp(alpha)+1
    ##           t dof     : alpha_dof = log(nu)
    ##=========================================================================

    ##=========================================================================
    def theta2alpha_cop(self, family, theta):
        """Native copula parameter -> unconstrained EM parameter (ParamTau.R)."""
        theta = np.asarray(theta, float)
        if family in ('gaussian', 't'):
            return 2.0 * np.arctanh(theta)          # log((1+rho)/(1-rho))
        elif family == 'clayton':
            return np.log(theta)
        elif family == 'frank':
            return theta
        elif family == 'gumbel':
            return np.log(theta - 1.0)
        raise ValueError("unknown family " + str(family))

    ##=========================================================================
    def alpha2theta_cop(self, family, alpha):
        """Unconstrained EM parameter -> native copula parameter (ParamCop.R).
        gaussian/t keep the R clamp |alpha| <= log(19999), i.e. |rho| <= 0.9999."""
        alpha = np.asarray(alpha, float)
        if family in ('gaussian', 't'):
            a = np.clip(alpha, -np.log(19999.0), np.log(19999.0))
            return np.tanh(a / 2.0)                 # 2 e^a/(e^a+1) - 1, safely
        elif family == 'clayton':
            return np.exp(alpha)
        elif family == 'frank':
            return alpha
        elif family == 'gumbel':
            return np.exp(alpha) + 1.0
        raise ValueError("unknown family " + str(family))

    ##=========================================================================
    def alpha2tau(self, family, alpha):
        """Unconstrained EM parameter -> Kendall tau (KendallTau.R).
        Frank uses tau = 1 - 4/a + 4*dilog(e^-a)/a^2 with
        dilog(x) = int_1^x log(t)/(1-t) dt = scipy.special.spence(x) exactly."""
        alpha = np.asarray(alpha, float)
        if family in ('gaussian', 't'):
            return (2.0 / np.pi) * np.arcsin(np.tanh(alpha / 2.0))
        elif family == 'clayton':
            theta = np.exp(alpha)
            return theta / (theta + 2.0)
        elif family == 'gumbel':
            return 1.0 / (1.0 + np.exp(-alpha))     # 1 - 1/theta
        elif family == 'frank':
            def tau_one(a):
                if abs(a) < 1e-5:
                    return a / 9.0                  # series limit at theta -> 0
                s = np.sign(a)
                a = abs(a)                          # tau is odd in alpha
                # e^-a underflows to 0 for large a; spence(0) = pi^2/6 is the
                # exact limit, so no special case is needed
                return s * (1.0 - 4.0 / a + 4.0 * spence(np.exp(-a)) / a ** 2)
            if alpha.ndim == 0:
                return tau_one(float(alpha))
            return np.array([tau_one(a) for a in alpha.ravel()]).reshape(alpha.shape)
        raise ValueError("unknown family " + str(family))

    ##=========================================================================
    def copula_logpdf(self, family, u, alpha, alpha_dof=None):
        """
        LOG density of the bivariate copula at pseudo-observations u (n x 2),
        parameterized by the UNCONSTRAINED alpha (copulaFamiliesPDF.R, d=2).
        R's contract is kept: rows with any u <= 0 or u >= 1 get density 0
        (log-density -inf). The t copula needs alpha_dof (nu = exp(alpha_dof)).
        Ported from the R formulas (frank cosh form, gumbel vmax/vmin trick,
        t in log space); the broken 1.1.0 PD check is NOT ported -- for d = 2
        with |rho| <= 0.9999 the correlation matrix is always PD.
        """
        from scipy import stats as sps
        from scipy.special import gammaln, ndtri

        u = np.asarray(u, float)
        u1, u2 = u[:, 0], u[:, 1]
        bad = (u1 <= 0) | (u1 >= 1) | (u2 <= 0) | (u2 >= 1)
        # evaluate at a safe point, overwrite with -inf afterwards (R: 0.5)
        u1 = np.where(bad, 0.5, u1)
        u2 = np.where(bad, 0.5, u2)

        if family == 'gaussian':
            rho = float(self.alpha2theta_cop('gaussian', alpha))
            x1, x2 = ndtri(u1), ndtri(u2)
            r2 = 1.0 - rho ** 2
            logc = (-0.5 * np.log(r2)
                    - (rho ** 2 * (x1 ** 2 + x2 ** 2) - 2.0 * rho * x1 * x2)
                    / (2.0 * r2))

        elif family == 't':
            if alpha_dof is None:
                raise ValueError("alpha_dof is required for the t copula")
            rho = float(self.alpha2theta_cop('t', alpha))
            nu = float(np.exp(alpha_dof))
            t1, t2 = sps.t.ppf(u1, nu), sps.t.ppf(u2, nu)
            r2 = 1.0 - rho ** 2
            quad = (t1 ** 2 - 2.0 * rho * t1 * t2 + t2 ** 2) / r2
            logc = (gammaln((nu + 2.0) / 2.0) + gammaln(nu / 2.0)
                    - 2.0 * gammaln((nu + 1.0) / 2.0) - 0.5 * np.log(r2)
                    - ((nu + 2.0) / 2.0) * np.log1p(quad / nu)
                    + ((nu + 1.0) / 2.0) * (np.log1p(t1 ** 2 / nu)
                                            + np.log1p(t2 ** 2 / nu)))

        elif family == 'clayton':
            a = float(alpha)
            if np.isinf(a) and a < 0:            # theta -> 0: independence
                logc = np.zeros(len(u1))
            else:
                theta = np.exp(a)
                # log(u1^-t + u2^-t - 1) via the max-exponent trick
                e1, e2 = -theta * np.log(u1), -theta * np.log(u2)
                m = np.maximum(e1, e2)
                logS = m + np.log(np.exp(e1 - m) + np.exp(e2 - m)
                                  - np.exp(-m))
                logc = (np.log1p(theta) - ((2.0 * theta + 1.0) / theta) * logS
                        - (theta + 1.0) * (np.log(u1) + np.log(u2)))

        elif family == 'frank':
            a = float(alpha)
            if abs(a) < 1e-10:                   # theta = 0: independence
                logc = np.zeros(len(u1))
            else:
                # D = 2 cosh(a(u2-u1)/2) - e^{a(u1+u2-2)/2} - e^{-a(u1+u2)/2}
                dlt, s = u2 - u1, u1 + u2
                exps = np.stack([a * dlt / 2.0, -a * dlt / 2.0,
                                 a * (s - 2.0) / 2.0, -a * s / 2.0])
                m = exps.max(axis=0)
                D = (np.exp(exps[0] - m) + np.exp(exps[1] - m)
                     - np.exp(exps[2] - m) - np.exp(exps[3] - m))
                logD = m + np.log(D)
                # a(1 - e^-a) > 0 for every a != 0
                log_num = (np.log(abs(a))
                           + (np.log1p(-np.exp(-a)) if a > 0
                              else -a + np.log1p(-np.exp(a))))
                logc = log_num - 2.0 * logD

        elif family == 'gumbel':
            a = float(alpha)
            if np.isinf(a) and a < 0:            # theta = 1: independence
                logc = np.zeros(len(u1))
            else:
                theta = np.exp(a) + 1.0
                v1, v2 = -np.log(u1), -np.log(u2)
                vmax, vmin = np.maximum(v1, v2), np.minimum(v1, v2)
                s = vmax * (1.0 + (vmin / vmax) ** theta) ** (1.0 / theta)
                logc = (np.log(theta - 1.0 + s) - s
                        + (theta - 1.0) * (np.log(v1) + np.log(v2))
                        + (v1 + v2) + (1.0 - 2.0 * theta) * np.log(s))

        else:
            raise ValueError("unknown family " + str(family))

        return np.where(bad, -np.inf, logc)

    ##=========================================================================
    def tau2alpha_unc(self, family, tau):
        """Kendall tau -> unconstrained EM parameter (ParamTau.R: iTau then
        log-transforms). Closed form except frank (brentq on alpha2tau, using
        oddness for tau < 0). Used for the block-wise EM initialization, where
        tau is clamped to [0.1, 0.9] as in EstHMMCop.R."""
        if family in ('gaussian', 't'):
            rho = np.sin(np.pi * tau / 2.0)
            return 2.0 * np.arctanh(rho)
        elif family == 'clayton':
            return np.log(2.0 * tau / (1.0 - tau))
        elif family == 'gumbel':
            return np.log(tau / (1.0 - tau))        # log(theta - 1)
        elif family == 'frank':
            if abs(tau) < 1e-8:
                return 9.0 * tau          # series inverse of tau ~ alpha/9
            s, t = np.sign(tau), abs(tau)
            a = brentq(lambda x: self.alpha2tau('frank', x) - t, 1e-10, 1e3)
            return s * a
        raise ValueError("unknown family " + str(family))

    ##=========================================================================
    def EMStep_cop(self, u, family, alpha, Q, optimizer_alg='Nelder-Mead'):
        """
        One EM iteration for the regime-switching copula (EstHMMCop.R::EMStep).
        alpha: unconstrained parameter vector, length reg (reg+1 for 't', the
        last entry being log(dof), SHARED by all regimes). Returns
        (nu_EM, alpha_new, Qnew, eta_EM, eta_bar_EM, lambda_EM, Lambda_EM, LL).
        """
        from scipy.optimize import minimize

        u = np.asarray(u, float)
        n = u.shape[0]
        r = Q.shape[0]
        is_t = (family == 't')
        a_dof = alpha[r] if is_t else None

        ## densities per regime (floored so the recursions never hit a 0 row)
        f = np.empty((n, r))
        for j in range(r):
            f[:, j] = np.exp(self.copula_logpdf(family, u, alpha[j], a_dof))
        f = np.maximum(f, 1e-300)

        ## backward: eta_bar (normalized each step, as in R)
        eta_bar = np.zeros((n, r))
        eta_bar[n - 1, :] = 1.0 / r
        for i in range(n - 2, -1, -1):
            v = (eta_bar[i + 1, :] * f[i + 1, :]) @ Q.T
            eta_bar[i, :] = v / v.sum()

        ## forward: eta, with LL from the normalizers
        eta = np.zeros((n, r))
        eta0 = np.ones(r) / r
        v = (eta0 @ Q) * f[0, :]
        Z = v.sum()
        LL = np.log(Z)
        eta[0, :] = v / Z
        for i in range(1, n):
            v = (eta[i - 1, :] @ Q) * f[i, :]
            Z = v.sum()
            LL += np.log(Z)
            eta[i, :] = v / Z

        ## smoothed lambda
        w = eta * eta_bar
        lam = w / w.sum(axis=1, keepdims=True)

        ## pairwise Lambda and Q update
        gc = eta_bar * f
        Qnew = np.zeros_like(Q)
        M = Q * np.outer(eta0, gc[0, :])
        Qnew += M / M.sum()
        for i in range(1, n):
            M = Q * np.outer(eta[i - 1, :], gc[i, :])
            Qnew += M / M.sum()
        Qnew = Qnew / Qnew.sum(axis=1, keepdims=True)

        nu_EM = lam.mean(axis=0)

        ## M-step: ONE joint optimization of the whole alpha vector
        ## (the t dof is shared across regimes, so regimes are coupled)
        def negloglik(av):
            tot = 0.0
            ad = av[r] if is_t else None
            for j in range(r):
                lp = self.copula_logpdf(family, u, av[j], ad)
                tot -= lam[:, j] @ np.maximum(lp, -700.0)
            return tot

        method = optimizer_alg if r >= 2 else 'L-BFGS-B'
        ## tolerances matched to R optim's Nelder-Mead (reltol ~1.5e-8)
        opts = ({'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 500}
                if method == 'Nelder-Mead' else None)
        res = minimize(negloglik, np.asarray(alpha, float), method=method,
                       options=opts)
        alpha_new = res.x

        return nu_EM, alpha_new, Qnew, eta, eta_bar, lam, None, LL

    ##=========================================================================
    def EstHMMCop(self, y, reg, family, max_iter=10000, ninit=100, eps=1e-4,
                  optimizer_alg='Nelder-Mead', initial_Q=None, initial_tau=None):
        """
        EM estimation of the bivariate Markov regime-switching copula model --
        Python port of HMMcopula::EstHMMCop (estimation only, no GoF).

        :param y: (n x 2) raw data (ranked internally to pseudo-observations)
        :param reg: number of regimes
        :param family: 'gaussian', 't', 'clayton', 'frank', 'gumbel'
        :param max_iter: cap on EM iterations after the warm-up
        :param ninit: warm-up EM iterations before convergence checking (R: 100)
        :param eps: stopping tolerance -- sum|d alpha| < sum|alpha| * (2*reg) * eps
                    (R uses r = reg*d = 2*reg; kept R-faithful, documented)
        :param initial_Q: starting transition matrix (default uniform, as R)
        :param initial_tau: starting Kendall tau per regime (default: block-wise
                    empirical tau clamped to [0.1, 0.9], as R)
        :return: dict with theta (native params), dof (t only), tau, Q, eta_EM,
                 nu_EM, lambda_EM, LL, AIC, BIC
        """
        from scipy.stats import rankdata, kendalltau

        y = np.asarray(y, float)
        n = y.shape[0]

        ## pseudo-observations, exactly R: floor(rank)/(n+1), average ranks
        u = np.floor(rankdata(y, axis=0, method='average')) / (n + 1.0)

        ## initialization: block-wise empirical Kendall tau, clamped [0.1, 0.9]
        if initial_tau is None:
            n0 = n // reg
            initial_tau = np.empty(reg)
            for j in range(reg):
                blk = u[j * n0:(j + 1) * n0, :]
                initial_tau[j] = np.clip(kendalltau(blk[:, 0], blk[:, 1])[0],
                                         0.1, 0.9)
        alpha0 = np.array([self.tau2alpha_unc(family, t) for t in initial_tau])
        if family == 't':
            alpha0 = np.append(alpha0, np.log(5.0))   # shared dof, R init

        Q0 = np.ones((reg, reg)) / reg if initial_Q is None \
            else np.asarray(initial_Q, float)

        ## warm-up (no convergence checking), then the monitored loop
        for _ in range(ninit):
            nu_EM, alpha_new, Qnew, eta, eta_bar, lam, _, LL = \
                self.EMStep_cop(u, family, alpha0, Q0, optimizer_alg)
            alpha0, Q0 = alpha_new, Qnew

        for _ in range(max_iter):
            nu_EM, alpha_new, Qnew, eta, eta_bar, lam, _, LL = \
                self.EMStep_cop(u, family, alpha0, Q0, optimizer_alg)
            sum1 = np.sum(np.abs(alpha0))
            sum2 = np.sum(np.abs(alpha_new - alpha0))
            alpha0, Q0 = alpha_new, Qnew
            if sum2 < sum1 * (2 * reg) * eps:
                break

        theta = self.alpha2theta_cop(family, alpha0[:reg])
        dof = float(np.exp(alpha0[reg])) if family == 't' else np.nan
        tau = np.array([self.alpha2tau(family, a) for a in alpha0[:reg]])

        numParam = reg + reg ** 2 + (1 if family == 't' else 0)
        out = {}
        out['theta'] = theta
        out['dof'] = dof
        out['tau'] = tau
        out['Q'] = Q0
        out['alpha'] = alpha0
        out['eta_EM'] = eta
        out['nu_EM'] = nu_EM
        out['lambda_EM'] = lam
        out['LL'] = LL
        out['AIC'] = (2 * numParam - 2 * LL) / n
        out['BIC'] = (np.log(n) * numParam - 2 * LL) / n
        return out
