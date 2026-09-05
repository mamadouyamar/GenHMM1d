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


class ARHMM:

    def SimMarkovChain(self, Q, n, eta0):
        """
        This function generates a Markov chain X(1), ..., X(n) with transition matrix Q, starting from a state eta0 or the uniform distribution on 1,..., r
        :param Q: transition matrix
        :param n: length of simulated time series
        :param eta0: nitial value in 1,...,r.
        :return: Markov chain
        """
        r,p = Q.shape

        x = np.zeros((n,1))
        x0 = np.zeros((n,r))

        ind = eta0

        if r > 1:
            for k in range(r):
                x0[0:n,k] = np.random.choice(r, n, p=Q[k,])
            for i in range(n):
                x[i] = x0[i][int(ind)]
                ind = int(x[i][0])
        else:
            x[0:n] = 0

        MC = x

        return(MC)


    ##=============================================================================
    ##=============================================================================
    ##=============================================================================
    ##=============================================================================
    def SimARXHMMGen(self, Q, theta, n, family='norm', burn_in=0):
        """
        This function simulates observation from a univariate hidden Markov model

        :param Q: transtion matrix
        :param family: 'norm', 'skewnorm'
        :param theta: parameters
        :param n: sample size (after burn-in removal)
        :param p: # of lags;
        :param burn_in: number of initial observations discarded
        :return: Simulated Hidden Markov Model
        """
        n = n + burn_in
        y = np.zeros(n)
        reg = Q.shape[0]
        sim = np.full((n,reg), np.nan)
        MC = np.zeros(n, dtype=int)

        if reg >= 2:
            MC = self.SimMarkovChain(Q=Q, n=n, eta0=1)
        else:
            MC[:(n+1)] = 1

        if family == 'norm':  ## [] ;     support [R]
            p = theta.shape[1] - 2
            for j in range(reg):
                _, a = theta.shape
                d = 0
                sim = np.zeros((n, reg))
                sim[0, :] = theta[int(MC[0][0]), 0] + np.random.randn() * theta[int(MC[0][0]), 1]
                y[:max(p, 1)] = sim[0, 0]

                for i in range(max(p, 1), n):
                    epsilon_i = np.random.normal(0, 1)
                    for j in range(reg):
                        if p > 0 and d == 0:
                            sim[i, j] = theta[j, 0] + np.dot(theta[j, 1:(p+1)], y[(i-p):i]) +\
                                        theta[j, -1] * epsilon_i
                        elif p == 0 and d == 0:
                            sim[i, j] = theta[j, 0] + theta[j, 1] * epsilon_i

                    y[i] = sim[i, int(MC[i][0])]

        #=================================== discard burn-in
        y   = y[burn_in:]
        sim = sim[burn_in:, :]
        MC  = MC[burn_in:]

        return (y, sim, MC)


    ##=============================================================================
    ##=============================================================================
    ##=============================================================================
    ##=============================================================================
    def SimARPoissonGen(self, Q, theta, n, burn_in=0):
        """
        This function simulates observation from a log-linear Poisson
        autoregressive hidden Markov model (paper M2):
            mu_t = exp(alpha_k + phi_k * log(1 + Y_{t-1})),   Y_t ~ Poisson(mu_t)

        :param Q: transtion matrix
        :param theta: parameters, one row [alpha, phi] per regime
        :param n: sample size (after burn-in removal)
        :param burn_in: number of initial observations discarded
        :return: Simulated Hidden Markov Model (y, sim, MC)
        """
        n = n + burn_in
        reg = Q.shape[0]
        theta = np.asarray(theta, dtype=float)

        MC = self.SimMarkovChain(Q=Q, n=n, eta0=1)
        y = np.zeros(n)
        sim = np.zeros((n, reg))

        for i in range(1, n):
            for j in range(reg):
                mu = np.exp(theta[j, 0] + theta[j, 1] * np.log(1.0 + y[i - 1]))
                sim[i, j] = np.random.poisson(mu)
            y[i] = sim[i, int(MC[i][0])]

        #=================================== discard burn-in
        y   = y[burn_in:]
        sim = sim[burn_in:, :]
        MC  = MC[burn_in:]

        return (y, sim, MC)


    ##=============================================================================
    ##=============================================================================
    ##=============================================================================
    ##=============================================================================
    def SimARZIGaussGen(self, Q, theta, n, burn_in=0):
        """
        This function simulates observation from a zero-inflated Gaussian
        autoregressive hidden Markov model (paper M3, zero-regime convention):
        REGIME 0 is a degenerate point mass at 0; regimes 1,...,reg-1 follow
            Y_t = c_k + phi_k * Y_{t-1} + sigma_k * eps_t.

        :param Q: transtion matrix
        :param theta: parameters (reg x 3); row 0 = zero regime (ignored),
                      rows 1.. = [c, phi, sigma]
        :param n: sample size (after burn-in removal)
        :param burn_in: number of initial observations discarded
        :return: Simulated Hidden Markov Model (y, sim, MC)
        """
        n = n + burn_in
        reg = Q.shape[0]
        theta = np.asarray(theta, dtype=float)

        MC = self.SimMarkovChain(Q=Q, n=n, eta0=1)
        y = np.zeros(n)
        sim = np.zeros((n, reg))

        for i in range(1, n):
            epsilon_i = np.random.normal(0, 1)      # shared shock, as in SimARXHMMGen
            sim[i, 0] = 0.0                          # zero regime
            for j in range(1, reg):
                sim[i, j] = theta[j, 0] + theta[j, 1] * y[i - 1] + \
                            theta[j, 2] * epsilon_i
            y[i] = sim[i, int(MC[i][0])]

        #=================================== discard burn-in
        y   = y[burn_in:]
        sim = sim[burn_in:, :]
        MC  = MC[burn_in:]

        return (y, sim, MC)


    ##=============================================================================
    ##=============================================================================
    ##=============================================================================
    ##=============================================================================
    def SimARZIPoisGen(self, Q, theta, n, burn_in=0):
        """
        This function simulates observation from a zero-inflated log-linear
        Poisson autoregressive hidden Markov model (paper M4, zero-regime
        convention): REGIME 0 is a degenerate point mass at 0; regimes 1,...
            mu_t = exp(alpha_k + phi_k * log(1 + Y_{t-1})),   Y_t ~ Poisson(mu_t)

        :param Q: transtion matrix
        :param theta: parameters (reg x 2); row 0 = zero regime (ignored),
                      rows 1.. = [alpha, phi]
        :param n: sample size (after burn-in removal)
        :param burn_in: number of initial observations discarded
        :return: Simulated Hidden Markov Model (y, sim, MC)
        """
        n = n + burn_in
        reg = Q.shape[0]
        theta = np.asarray(theta, dtype=float)

        MC = self.SimMarkovChain(Q=Q, n=n, eta0=1)
        y = np.zeros(n)
        sim = np.zeros((n, reg))

        for i in range(1, n):
            sim[i, 0] = 0.0                          # zero regime
            for j in range(1, reg):
                mu = np.exp(theta[j, 0] + theta[j, 1] * np.log(1.0 + y[i - 1]))
                sim[i, j] = np.random.poisson(mu)
            y[i] = sim[i, int(MC[i][0])]

        #=================================== discard burn-in
        y   = y[burn_in:]
        sim = sim[burn_in:, :]
        MC  = MC[burn_in:]

        return (y, sim, MC)


    def dens_gauss(self, y, theta, Z_augmented):
        if len(Z_augmented.shape) == 1:
            n_Z = Z_augmented.shape[0]
            d_Z_aug = 1
        else:
            n_Z, d_Z_aug = Z_augmented.shape
            d_Z_aug += 1
        d_Z = d_Z_aug - 1
        n = len(y)
        mu = np.zeros(n)
        d_X = len(theta) - 1 - d_Z

        if d_Z_aug == 1:
            p = d_X - 1
            d_Z = 0

        if d_X > 1 and d_Z == 0:
            for i in range(p+1, n):
                mu[i] = np.dot(np.concatenate([[1], y[(i-p):i]]),
                               theta[:-1])
        elif d_X == 1:
            for i in range(p+1, n):
                mu[i] = theta[0]

        y_r = y[p:]
        mu_r = mu[p:]

        f = stats.norm.pdf(y_r, mu_r, np.exp(theta[-1]))

        return f


    def cum_gauss(self, y, theta, Z_augmented):
        if len(Z_augmented.shape) == 1:
            n_Z = Z_augmented.shape[0]
            d_Z_aug = 1
        else:
            n_Z, d_Z_aug = Z_augmented.shape
            d_Z_aug += 1
        d_Z = d_Z_aug - 1
        n = len(y)
        mu = np.zeros(n)
        d_X = len(theta) - 1 - d_Z

        if d_Z_aug == 1:
            p = d_X - d_Z_aug
            d_Z = 0

        if d_X > 1 and d_Z == 0:
            for i in range(p+1, n):
                mu[i] = np.dot(np.concatenate([[1], y[(i-p):i]]),
                               theta[:-1])
        elif d_X == 1:
            for i in range(p+1, n):
                mu[i] = theta[0]

        y_r = y[p:]
        mu_r = mu[p:]

        F = stats.norm.cdf(y_r, mu_r, theta[-1])

        return F




    def dens_pois(self, y, theta, Z_augmented):
        """
        Conditional density of the log-linear Poisson AR(1) regime (paper M2):
            mu_t = exp(alpha + phi * log(1 + y_{t-1})),   Y_t | y_{t-1} ~ Poisson(mu_t)
        theta = [alpha, phi]. Returns the pmf for t = 2,...,n (length n-1),
        floored at 1e-300 so the EM recursions and log-objectives stay finite.
        """
        eta = theta[0] + theta[1] * np.log1p(y[:-1])
        mu = np.exp(np.clip(eta, -30.0, 30.0))
        f = stats.poisson.pmf(np.round(y[1:]), mu)
        return np.maximum(f, 1e-300)


    ##=============================================================================
    def EMStep_ar(self, y, family, theta, Q, p_AR, Z, optimizer_alg, ZI=0):
        """
        This function perform EM optimization

        :param y: time series
        :param family: distribution name
        :param theta: parameters
        :param Q: transition matrix
        :param optimizer_alg: optmization algorithm. default ('Nelder-Mead')
        :param ZI: 1 if zero-inflated (regime 0 = point mass at 0, paper M3/M4), 0 otherwise
        :return:
        """

        n = len(y) - p_AR
        r, p = theta.shape
        eta_bar_EM = np.zeros((n,r))
        eta_EM = np.zeros((n,r))
        lambda_EM = np.zeros((n,r))
        Lambda_EM = np.zeros((n,r))
        f = np.zeros((n,r))
        Z_i = np.zeros((n,1))
        Lambda_EM = np.zeros((r,r,n))

        if family == 'norm':
            for j in range(ZI, r):
                f[0:n, j] = self.dens_gauss(y=y, theta=theta[j, :], Z_augmented=Z)

        elif family == 'poisson':
            for j in range(ZI, r):
                f[0:n, j] = self.dens_pois(y=y, theta=theta[j, :], Z_augmented=Z)

        if ZI == 1:
            ## regime 0 = point mass at 0 (paper M3/M4): g_0(y) = 1(y = 0)
            zero_mask = (np.abs(y[p_AR:]) < 1e-12)
            f[0:n, 0] = zero_mask.astype(float)
            if family == 'norm':
                ## M3: the zero regime is observable -- a continuous regime
                ## cannot produce an exact zero, so its density is 0 there
                f[zero_mask, 1:] = 0.0
            ## M4 (poisson): regime 0 is NOT observable -- Poisson regimes also
            ## emit zeros, so their pmf at y = 0 is kept as is

        ## eta_bar_EM
        eta_bar_EM[n-1,0:r] = 1/r
        for k in range(n-1):
            i = n-2-k
            j = i+1
            v = np.multiply(eta_bar_EM[j,0:r], f[j,0:r]).dot(np.transpose(Q))
            eta_bar_EM[i,0:r] = v/sum(v)

        ## eta_EM
        eta0 = np.ones((1,r))/r
        v = np.multiply( ( eta0.dot(Q) ), f[0,0:r])
        Z_i[0] = sum(sum(v))
        eta_EM[0,0:r] = v/Z_i[0]

        for i in range(1,n):
            v = np.multiply( ( eta_EM[i-1,0:r].dot(Q) ), f[i,0:r] )
            Z_i[i] = sum(v)
            eta_EM[i,0:r] = v/Z_i[i]

        LL = sum(np.log(Z_i))

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
        for i in range(ZI, r):
            ## CA PREND TROP DE TEMPS SANS DOUTE A CAUSE DE LA FONCTION LAMBDA ??
            ## M-step (paper Appendix A.2): beta_j = argmax sum_t lambda_t(j) log g_{j,beta_j}
            if family == 'norm':
                fun = lambda thetaa : -lambda_EM[0:n, i] @ np.log(np.maximum(self.dens_gauss(y, thetaa, Z), 1e-300)).T
            elif family == 'poisson':
                fun = lambda thetaa : -lambda_EM[0:n, i] @ np.log(self.dens_pois(y, thetaa, Z)).T

            if optimizer_alg == 'Nelder-Mead':
                res = minimize(fun, theta[i,0:p], method='Nelder-Mead')  # 'Nelder-Mead'
            elif optimizer_alg == 'CG':
                res = minimize(fun, theta[i, 0:p], method='CG')
            elif optimizer_alg == 'BFGS':
                res = minimize(fun, theta[i, 0:p], method='BFGS')
            elif optimizer_alg == 'L-BFGS-B':
                res = minimize(fun, theta[i, 0:p], method='L-BFGS-B')

            theta_new_EM[i,0:p] = res.x

        return (nu_EM, theta_new_EM, Qnew_EM, eta_EM, eta_bar_EM, lambda_EM, Lambda_EM, LL)



    ##=============================================================================
    def Sn1d(self, U):
        """
        his function computes the Cramer-von Mises statistic Sn for goodness-of-fit of the null hypothesis of a univariate uniform distrubtion over [0,1]

        :param U: vector of pseudos-observations (approximating uniform)
        :return: Cramer-von Mises statistic
        """
        n = len(U)
        u = np.sort(U)
        t = (-0.5 + np.arange(1, n+1) ) / n

        stat = (1/(12*n)) + sum( (u-t)**2 )

        return(stat)




    ##=============================================================================
    def EstHMMGen_AR(self, y, reg, family='norm', p_AR=1, percentiles=None, max_iter=10000, ninit=20, eps=10e-20,
                     optimizer_alg='Nelder-Mead', init_rand=False, initial_Q=None, initial_theta=None, ZI=0):
        """
        EM estimation of autoregressive HMMs, following Nasri, Remillard &
        Thioub (2024, JSCS) Appendix 1: E-step recursions (A1)-(A5), M-step
        beta_j = argmax sum_t lambda_t(j) log g_{j,beta_j}(y_t | y_{t-1}).
        Covered models (paper section 3.1):
            M1: family='norm'               theta rows [c, phi_1..phi_p, sigma]
            M2: family='poisson'            theta rows [alpha, phi] (log-linear)
            M3: family='norm',    ZI=1      regime 0 = point mass at 0 (observable)
            M4: family='poisson', ZI=1      regime 0 = point mass at 0 (hidden)

        :param y: time series
        :param reg: number of regimes (including the zero regime when ZI=1)
        :param family: distribution name ('norm' or 'poisson')
        :param percentiles: used to calibrate the initial parameters
        :param max_iter: maximum number of iteration
        :param ninit: minimin number of iteration
        :param eps: tolerance paramets
        :param optimizer_alg: optimizer_alg: optmization algorithm. default ('Nelder-Mead')
        :param init_rand: use to calibrate the initial parameters
        :param ZI: 1 if zero-inflated (regime 0 = point mass at 0), 0 otherwise
        :return: estimated HMM
        """

        if isinstance(percentiles, list):
            reg = len(percentiles) + 1 + ZI
            percentiles = [1e-9] + percentiles + [100]

        n = len(y)

        ## data used to initialize the emission parameters: with ZI=1 the zero
        ## regime is pinned at row 0, so init uses the non-zero observations
        y_init = y[np.abs(y) > 1e-12] if ZI == 1 else y
        ng = reg - ZI                          # number of non-zero regimes
        n0 = math.floor(len(y_init)/ng)
        ind0 = np.arange(n0)

        if family == 'norm':
            p = p_AR + 2                       # [c, phi_1..phi_p, log sigma]
            theta_init = np.asarray([np.mean(y_init), *np.full(p_AR, 0.5),
                                     np.log(np.std(y_init))])
            dens = self.dens_gauss
        elif family == 'poisson':
            p = p_AR + 1                       # [alpha, phi] (log-linear, M2)
            theta_init = np.asarray([np.log(np.mean(y_init) + 1e-9),
                                     *np.full(p_AR, 0.1)])
            dens = self.dens_pois

        theta0 = np.zeros((reg, p))
        alpha0 = np.zeros((reg, p))            # ZI: row 0 stays 0 (point mass)

        if initial_theta is None:
            for j in range(ZI, reg):
                jj = j - ZI
                if percentiles == None or isinstance(percentiles, list) == False:
                    ind = jj*n0+ind0
                    if init_rand==False:
                        x = y_init[ind]
                    else:
                        x = y_init[np.random.choice(len(y_init), int(np.round(0.8*len(y_init))), replace=False)]
                else:
                    x = y_init[(y_init > np.percentile(y_init, percentiles[jj])) &
                               (y_init <= np.percentile(y_init, percentiles[jj+1]))]

                X_temp = np.ones(len(x))
                fun = lambda thetaa: -np.sum(np.log(np.maximum(dens(x, thetaa, X_temp), 1e-300)))
                res = minimize(fun, theta_init, method=optimizer_alg)
                alpha0[j, 0:p] = res.x.copy()

        else:
            theta0 = initial_theta
            for j in range(ZI, reg):
                temp_fit = theta0[j, :].copy()
                if family == 'norm':
                    temp_fit[-1] = np.log(temp_fit[-1])
                alpha0[j, :] = temp_fit

        if initial_Q is None:
            Q0 = np.ones((reg, reg))/reg
        else:
            Q0 = initial_Q

        Z = np.ones(n)

        for k in range(ninit):
            nu_EM, alpha_new_EM, Qnew_EM, eta_EM, eta_bar_EM, lambda_EM, Lambda_EM, LL =\
                self.EMStep_ar(y=y, family=family, theta=alpha0, Q=Q0, p_AR=p_AR, Z=Z, optimizer_alg=optimizer_alg, ZI=ZI)
            Q0 = Qnew_EM
            alpha0 = alpha_new_EM

        for k in range(max_iter):
            nu_EM, alpha_new_EM, Qnew_EM, eta_EM, eta_bar_EM, lambda_EM, Lambda_EM, LL =\
                self.EMStep_ar(y=y, family=family, theta=alpha0, Q=Q0, p_AR=p_AR, Z=Z, optimizer_alg=optimizer_alg, ZI=ZI)
            sum1 = sum(sum(abs(alpha0)))
            sum2 = sum(sum(abs(alpha_new_EM-alpha0)))
            if (sum2 < sum1 * reg * eps):
                break
            Q0 = Qnew_EM
            alpha0 = alpha_new_EM


        alpha = alpha_new_EM
        Q = Qnew_EM

        theta = np.zeros((reg, p))
        for j in range(ZI, reg):
            t_alpha = alpha[j, :].copy()
            if family == 'norm':
                t_alpha[-1] = np.exp(t_alpha[-1])   # log sigma -> sigma
            theta[j, :] = t_alpha.copy()

        t_mean_s = np.zeros((reg))
        for j in range(ZI, reg):
            if family == 'norm':
                if p_AR == 1:
                    t_mean_s[j] = theta[j, 0] / (1-theta[j, 1:-1].sum())
                elif p_AR == 0:
                    t_mean_s[j] = theta[j, 0]
            elif family == 'poisson':
                ## predicted regime mean at the average lag level (the
                ## log-linear AR-Poisson has no closed-form stationary mean)
                t_mean_s[j] = np.exp(theta[j, 0] + theta[j, 1] * np.mean(np.log1p(y)))

        ## label alignment: the zero regime (if any) is pinned first, the
        ## remaining regimes are sorted by their (stationary/predicted) mean
        order = np.concatenate([np.arange(ZI),
                                ZI + np.argsort(t_mean_s[ZI:])]).astype(int)
        t_mean_s = t_mean_s[order]
        theta = theta[order, :]
        temp_Q = Q[order,:]
        Q = temp_Q[:,order]
        eta_EM = eta_EM[:, order]
        lambda_EM = lambda_EM[:, order]
        # Lambda_EM = Lambda_EM[:, order]

        numel = (reg - ZI)*theta.shape[1]      # zero regime has no parameters
        numParam = (numel+reg**2)

        AIC = (2 * numParam - 2 * LL) / n
        BIC = (np.log(n) * numParam - 2 * LL) / n
        CAIC = ((np.log(n)+1) * numParam - 2 * LL) / n
        AICc = AIC + (2*numParam*(numParam+1))/(n-numParam-1)
        HQC = (2 * numParam*np.log(np.log(n)) - 2 * LL)/n

        ## pseudo-observations / CvM statistic: continuous, non-ZI case only
        if family == 'norm' and ZI == 0:
            cdf_gof = np.zeros((n-p_AR, reg))
            for j in range(reg):
                cdf_gof[:,j] = self.cum_gauss(y, theta[j,:], Z)

            eta00 = np.ones((1,reg))/reg
            w00 = np.concatenate((eta00, eta_EM), axis=0)
            W = w00[0:n,0:reg].dot(Q)
            W = W[:(n-p_AR), :].copy()
            U = np.sum( np.multiply(W, cdf_gof), -1 )
            cvm = self.Sn1d(U=U)
        else:
            W = np.full((n-p_AR, reg), np.nan)
            U = np.full(n-p_AR, np.nan)
            cvm = np.nan

        pred_e = np.zeros((n-p_AR, 1))
        pred_l = np.zeros((n-p_AR, 1))
        for i in range(len(pred_e)):
            bb_e = np.argsort(eta_EM[i, 0:reg])
            pred_e[i] = bb_e[-1]
            bb_l = np.argsort(lambda_EM[i, 0:reg])
            pred_l[i] = bb_l[-1]
        pred_e = pred_e+1
        pred_l = pred_l+1

        statistics = np.zeros((reg,2))
        mean_s = np.zeros((reg,1))
        var_s = np.zeros((reg,1))
        for j in range(ZI, reg):
            if family == 'norm':
                if p_AR == 1:
                    mean_s[j] = theta[j, 0] / (1 - theta[j, 1:-1].sum())
                    var_s[j] = (theta[j, -1]**2) / (1 - theta[j,1]**2)
                elif p_AR == 0:
                    mean_s[j] = theta[j, 0]
                    var_s[j] = theta[j, -1] ** 2
            elif family == 'poisson':
                mean_s[j] = t_mean_s[j]
                var_s[j] = t_mean_s[j]         # Poisson: variance = mean

        statistics[0:reg, 0] = mean_s.squeeze(-1)
        statistics[0:reg, 1] = var_s.squeeze(-1)**0.5


        time_in_each_reg = np.linalg.matrix_power(Q, 1000)[0, :]

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
        out['time_in_each_reg'] = time_in_each_reg

        return out



